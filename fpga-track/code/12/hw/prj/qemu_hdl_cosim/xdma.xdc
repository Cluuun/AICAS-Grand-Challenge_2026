# ===========================================================
# XDMA PCIe Gen4 x8 约束  (XCVU37P-fsvh2892)
# XDMA 配置: AXI Bridge, Advanced, x8, 16 GT/s
#   Reference Clock = 100 MHz, AXI Clock = 250 MHz
#   AXI Addr Width = 64, AXI Data Width = 512-bit
# ===========================================================

# 时钟约束: PCIe 参考时钟 100 MHz (period = 10 ns)
create_clock -name sys_clk -period 10 [get_ports {pcie_refclk_clk_p[0]}]

# 参考时钟引脚 (MGTREFCLK)
set_property PACKAGE_PIN AR15 [get_ports {pcie_refclk_clk_p[0]}]
set_property PACKAGE_PIN AR14 [get_ports {pcie_refclk_clk_n[0]}]

# 复位信号 (PCIe PERST#, active-low, 异步)
set_property PACKAGE_PIN BF5       [get_ports pcie_perstn]
set_property IOSTANDARD  LVCMOS18  [get_ports pcie_perstn]
set_property PULLUP      true      [get_ports pcie_perstn]

# 复位信号时序例外: 异步复位不参与同步时序分析
set_false_path -from [get_ports pcie_perstn]

# RXN 通道 (去掉方括号内的空格)
set_property PACKAGE_PIN AL1  [get_ports {pci_express_rxn[0]}]
set_property PACKAGE_PIN AM3  [get_ports {pci_express_rxn[1]}]
set_property PACKAGE_PIN AN5  [get_ports {pci_express_rxn[2]}]
set_property PACKAGE_PIN AN1  [get_ports {pci_express_rxn[3]}]
set_property PACKAGE_PIN AP3  [get_ports {pci_express_rxn[4]}]
set_property PACKAGE_PIN AR1  [get_ports {pci_express_rxn[5]}]
set_property PACKAGE_PIN AT3  [get_ports {pci_express_rxn[6]}]
set_property PACKAGE_PIN AU1  [get_ports {pci_express_rxn[7]}]
#set_property PACKAGE_PIN AV3  [get_ports {pci_express_rxn[8]}]
#set_property PACKAGE_PIN AW5  [get_ports {pci_express_rxn[9]}]
#set_property PACKAGE_PIN AW1  [get_ports {pci_express_rxn[10]}]
#set_property PACKAGE_PIN AY3  [get_ports {pci_express_rxn[11]}]
#set_property PACKAGE_PIN BA5  [get_ports {pci_express_rxn[12]}]
#set_property PACKAGE_PIN BA1  [get_ports {pci_express_rxn[13]}]
#set_property PACKAGE_PIN BB3  [get_ports {pci_express_rxn[14]}]
#set_property PACKAGE_PIN BC1  [get_ports {pci_express_rxn[15]}]

# RXP 通道
set_property PACKAGE_PIN AL2  [get_ports {pci_express_rxp[0]}]
set_property PACKAGE_PIN AM4  [get_ports {pci_express_rxp[1]}]
set_property PACKAGE_PIN AN6  [get_ports {pci_express_rxp[2]}]
set_property PACKAGE_PIN AN2  [get_ports {pci_express_rxp[3]}]
set_property PACKAGE_PIN AP4  [get_ports {pci_express_rxp[4]}]
set_property PACKAGE_PIN AR2  [get_ports {pci_express_rxp[5]}]
set_property PACKAGE_PIN AT4  [get_ports {pci_express_rxp[6]}]
set_property PACKAGE_PIN AU2  [get_ports {pci_express_rxp[7]}]
#set_property PACKAGE_PIN AV4  [get_ports {pci_express_rxp[8]}]
#set_property PACKAGE_PIN AW6  [get_ports {pci_express_rxp[9]}]
#set_property PACKAGE_PIN AW2  [get_ports {pci_express_rxp[10]}]
#set_property PACKAGE_PIN AY4  [get_ports {pci_express_rxp[11]}]
#set_property PACKAGE_PIN BA6  [get_ports {pci_express_rxp[12]}]
#set_property PACKAGE_PIN BA2  [get_ports {pci_express_rxp[13]}]
#set_property PACKAGE_PIN BB4  [get_ports {pci_express_rxp[14]}]
#set_property PACKAGE_PIN BC2  [get_ports {pci_express_rxp[15]}]

# TXN 通道
set_property PACKAGE_PIN AL10 [get_ports {pci_express_txn[0]}]
set_property PACKAGE_PIN AM8  [get_ports {pci_express_txn[1]}]
set_property PACKAGE_PIN AN10 [get_ports {pci_express_txn[2]}]
set_property PACKAGE_PIN AP8  [get_ports {pci_express_txn[3]}]
set_property PACKAGE_PIN AR10 [get_ports {pci_express_txn[4]}]
set_property PACKAGE_PIN AR6  [get_ports {pci_express_txn[5]}]
set_property PACKAGE_PIN AT8  [get_ports {pci_express_txn[6]}]
set_property PACKAGE_PIN AU10 [get_ports {pci_express_txn[7]}]
#set_property PACKAGE_PIN AU6  [get_ports {pci_express_txn[8]}]
#set_property PACKAGE_PIN AV8  [get_ports {pci_express_txn[9]}]
#set_property PACKAGE_PIN AW10 [get_ports {pci_express_txn[10]}]
#set_property PACKAGE_PIN AY8  [get_ports {pci_express_txn[11]}]
#set_property PACKAGE_PIN BA10 [get_ports {pci_express_txn[12]}]
#set_property PACKAGE_PIN BB8  [get_ports {pci_express_txn[13]}]
#set_property PACKAGE_PIN BC10 [get_ports {pci_express_txn[14]}]
#set_property PACKAGE_PIN BC6  [get_ports {pci_express_txn[15]}]

# TXP 通道
set_property PACKAGE_PIN AL11 [get_ports {pci_express_txp[0]}]
set_property PACKAGE_PIN AM9  [get_ports {pci_express_txp[1]}]
set_property PACKAGE_PIN AN11 [get_ports {pci_express_txp[2]}]
set_property PACKAGE_PIN AP9  [get_ports {pci_express_txp[3]}]
set_property PACKAGE_PIN AR11 [get_ports {pci_express_txp[4]}]
set_property PACKAGE_PIN AR7  [get_ports {pci_express_txp[5]}]
set_property PACKAGE_PIN AT9  [get_ports {pci_express_txp[6]}]
set_property PACKAGE_PIN AU11 [get_ports {pci_express_txp[7]}]
#set_property PACKAGE_PIN AU7  [get_ports {pci_express_txp[8]}]
#set_property PACKAGE_PIN AV9  [get_ports {pci_express_txp[9]}]
#set_property PACKAGE_PIN AW11 [get_ports {pci_express_txp[10]}]
#set_property PACKAGE_PIN AY9  [get_ports {pci_express_txp[11]}]
#set_property PACKAGE_PIN BA11 [get_ports {pci_express_txp[12]}]
#set_property PACKAGE_PIN BB9  [get_ports {pci_express_txp[13]}]
#set_property PACKAGE_PIN BC11 [get_ports {pci_express_txp[14]}]
#set_property PACKAGE_PIN BC7  [get_ports {pci_express_txp[15]}]

