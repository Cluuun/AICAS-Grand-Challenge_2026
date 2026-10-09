module weight_feeder_top
  import npu_config_pkg::*;
#(
    parameter int RF_DATA_WIDTH      = npu_config_pkg::RF_DATA_WIDTH,
    parameter int PE_DATA_WIDTH_IN   = npu_config_pkg::PE_DATA_WIDTH_IN,
    parameter int ARRAY_WIDTH        = npu_config_pkg::ARRAY_WIDTH,
    parameter int ARRAY_HEIGHT       = npu_config_pkg::ARRAY_HEIGHT,
    parameter int SPM_SIZE           = npu_config_pkg::SPM_SIZE,
    parameter int SPM_ADDR_WIDTH     = $clog2(SPM_SIZE)
)
(
    input  logic                                          clk,
    input  logic                                          rst_n,

    //------------------------------------------
    // Control
    //------------------------------------------
    input  logic [3:0]                                    cfg_compute_weight_shape_m1,
    input  logic [RF_DATA_WIDTH/2-1:0]                    sa_input_weight_spm_addr,
    input  logic [$clog2(ARRAY_WIDTH)-1:0]                cfg_compute_weight_channel_out_m1,  // can be less than ARRAY_WIDTH-1
    input  logic [$clog2(ARRAY_HEIGHT)-1:0]               cfg_compute_weight_channel_in_m1,   // can be less than ARRAY_HEIGHT-1

    //------------------------------------------
    // Scratchpad
    //------------------------------------------
    output logic                                          weight_spm_rd_en,
    output logic [SPM_ADDR_WIDTH-1:0]                     weight_spm_rd_addr,
    input  logic [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0]  weight_spm_rd_data_in,

    //------------------------------------------
    // Interface to systolic array
    //------------------------------------------
    input  logic                                          weight_feeder_sa_ready,
    output logic                                          weight_feeder_sa_valid,
    output logic [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0]  weight_feeder_sa_data,

    //------------------------------------------
    // Command handshake
    //------------------------------------------
    input  logic                                          weight_feeder_req_en,
    output logic                                          weight_feeder_busy,
    output logic                                          weight_feeder_done
);

  // Internal signals
  logic [4:0]                                       cfg_compute_weight_shape;
  logic [$clog2(ARRAY_WIDTH):0]                     cfg_compute_weight_channel_out;
  logic [$clog2(ARRAY_HEIGHT):0]                    cfg_compute_weight_channel_in;

  logic                                             any_fifo_full;
  logic [ARRAY_WIDTH-1:0]                           fifo_full;
  logic [ARRAY_WIDTH-1:0]                           fifo_empty;
  logic                                             fifo_wr_en;
  logic [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0]     fifo_data_in;
  logic [ARRAY_WIDTH-1:0]                           fifo_valid;
  logic [ARRAY_WIDTH-1:0]                           fifo_empty_raw;

  logic [RF_DATA_WIDTH/2-1:0]                       sa_input_weight_spm_addr_r;

  //------------------------------------------
  // Configuration registers
  //------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
      cfg_compute_weight_shape       <= 5'b0;
      cfg_compute_weight_channel_out <= '0;
      cfg_compute_weight_channel_in  <= '0;
      fifo_valid                     <= '1;
      sa_input_weight_spm_addr_r     <= '0;
    end else if (weight_feeder_req_en) begin
      cfg_compute_weight_shape       <= cfg_compute_weight_shape_m1 + 1;
      cfg_compute_weight_channel_out <= cfg_compute_weight_channel_out_m1 + 1;
      cfg_compute_weight_channel_in  <= cfg_compute_weight_channel_in_m1 + 1;
      sa_input_weight_spm_addr_r     <= sa_input_weight_spm_addr;
      for (int i = 0; i < ARRAY_WIDTH; i++) begin
        fifo_valid[i] <= (i <= cfg_compute_weight_channel_out_m1);
      end
    end
    else begin
      cfg_compute_weight_shape       <= cfg_compute_weight_shape;
      cfg_compute_weight_channel_out <= cfg_compute_weight_channel_out;
      cfg_compute_weight_channel_in  <= cfg_compute_weight_channel_in;
      fifo_valid                     <= fifo_valid;
      sa_input_weight_spm_addr_r     <= sa_input_weight_spm_addr_r;
    end
  end

  assign any_fifo_full = |fifo_full;

  //------------------------------------------
  // FSM States
  //------------------------------------------
  typedef enum logic[1:0] {
    IDLE,
    INIT,
    BUSY,
    DONE
  } state_t;

  state_t state, state_next;

  //------------------------------------------
  // FSM State Register
  //------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
      state <= IDLE;
    end else begin
      state <= state_next;
    end
  end

  // Decided by max value of cfg_compute_weight_channel_in * cfg_compute_weight_shape * cfg_compute_weight_shape
  logic [11:0] counter_kernel;
  logic [11:0] counter_kernel_send;

  //------------------------------------------
  // FSM Next State Logic
  //------------------------------------------
  always_comb begin
    state_next = state;

    case (state)
      IDLE: begin
        if (weight_feeder_req_en) begin
          state_next = INIT;
        end
      end

      INIT: begin
        state_next = BUSY;
      end

      BUSY: begin
        if ((~any_fifo_full) && 
            (counter_kernel >= ARRAY_HEIGHT * cfg_compute_weight_shape * cfg_compute_weight_shape)) begin
          state_next = DONE;
        end
      end

      DONE: begin
        if (&fifo_empty_raw) begin
          state_next = IDLE;
        end
      end

      default: begin
        state_next = IDLE;
      end
    endcase
  end

  //------------------------------------------
  // FSM Output Logic
  //------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
      weight_feeder_busy <= 1'b0;
      weight_feeder_done <= 1'b0;
    end else begin
      weight_feeder_busy <= (state == BUSY);
      weight_feeder_done <= (state == DONE);
    end
  end

  logic pipe1_valid;
  logic data_zero;
  logic [7:0] channel_block_count;

  //------------------------------------------
  // Pipeline Stage 1: Address Generation
  //------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
      counter_kernel      <= 12'b0;
      counter_kernel_send <= 12'b0;
      pipe1_valid         <= 1'b0;
      data_zero           <= 1'b0;
      channel_block_count <= 8'b0;
      weight_spm_rd_addr  <= '0;
    end else begin
      case (state)
        IDLE, DONE: begin
          counter_kernel      <= 12'b0;
          counter_kernel_send <= 12'b0;
          pipe1_valid         <= 1'b0;
          data_zero           <= 1'b0;
          channel_block_count <= 8'b0;
          weight_spm_rd_addr  <= '0;
        end

        INIT: begin
          counter_kernel      <= 12'b0;
          counter_kernel_send <= 12'b0;
          pipe1_valid         <= 1'b0;
          data_zero           <= 1'b0;
          channel_block_count <= 8'b0;
          weight_spm_rd_addr  <= sa_input_weight_spm_addr_r;
        end
        
        BUSY: begin
          if (~any_fifo_full) begin
            if (counter_kernel >= ARRAY_HEIGHT * cfg_compute_weight_shape * cfg_compute_weight_shape) begin
              counter_kernel      <= 12'b0;
              counter_kernel_send <= 12'b0;
              pipe1_valid         <= 1'b0;
              data_zero           <= 1'b0;
              channel_block_count <= 8'b0;
              weight_spm_rd_addr  <= sa_input_weight_spm_addr_r;
            end else begin
              counter_kernel      <= counter_kernel + 1;
              counter_kernel_send <= counter_kernel;
              pipe1_valid         <= 1'b1;
              data_zero           <= (channel_block_count >= cfg_compute_weight_channel_in);
              channel_block_count <= (channel_block_count == ARRAY_HEIGHT - 1) ? 8'b0 : (channel_block_count + 1);
              if (counter_kernel != 0) begin
                weight_spm_rd_addr  <= (channel_block_count >= cfg_compute_weight_channel_in) ? weight_spm_rd_addr : (weight_spm_rd_addr + cfg_compute_weight_channel_out * PE_DATA_WIDTH_IN / 8);
              end
              else begin
                weight_spm_rd_addr <= weight_spm_rd_addr;
              end
            end
          end else begin
            counter_kernel      <= counter_kernel;
            counter_kernel_send <= counter_kernel_send;
            pipe1_valid         <= 1'b0;
            data_zero           <= data_zero;
            channel_block_count <= channel_block_count;
            weight_spm_rd_addr  <= weight_spm_rd_addr;
          end
        end
      endcase
    end
  end

  assign weight_spm_rd_en   = pipe1_valid && (~data_zero); 

  logic data_zero_r;

  //------------------------------------------
  // Pipeline Stage 2: Data Buffering
  //------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
      fifo_wr_en <= 1'b0;
      data_zero_r <= 1'b0;
    end else begin
      fifo_wr_en <= pipe1_valid;
      data_zero_r <= data_zero;
    end
  end

  assign fifo_data_in = data_zero_r ? '0 : weight_spm_rd_data_in;

  //------------------------------------------
  // FIFO Instantiation
  //------------------------------------------
  logic [ARRAY_WIDTH-1:0][PE_DATA_WIDTH_IN-1:0] fifo_data_raw;

  genvar i;
  generate
    for (i = 0; i < ARRAY_WIDTH; i++) begin : w_fifo
      FIFO_DP_GT_1 #(
        .DW (PE_DATA_WIDTH_IN),
        .DP (4)
      ) u_weight_feeder_fifo (
        .clk       (clk),
        .rst_n     (rst_n),

        .fifo_wen  (fifo_wr_en && fifo_valid[i]),
        .fifo_wdat (fifo_data_in[i]),
        .fifo_full (fifo_full[i]),

        .fifo_ren  (weight_feeder_sa_ready && fifo_valid[i]),
        .fifo_rdat (fifo_data_raw[i]),
        .fifo_empty(fifo_empty_raw[i])
      );
      
      assign fifo_empty[i]            = fifo_valid[i] ? fifo_empty_raw[i] : 1'b0;
      assign weight_feeder_sa_data[i] = fifo_valid[i] ? fifo_data_raw[i] : '0;
    end
  endgenerate

  //------------------------------------------
  // Output to Systolic Array
  //------------------------------------------
  assign weight_feeder_sa_valid = ~(|fifo_empty);

endmodule
