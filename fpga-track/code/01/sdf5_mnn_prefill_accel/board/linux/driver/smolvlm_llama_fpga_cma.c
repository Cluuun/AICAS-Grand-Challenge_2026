// SPDX-License-Identifier: GPL-2.0
#include <linux/dma-mapping.h>
#include <linux/fs.h>
#include <linux/ioctl.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/slab.h>

#define SMOL_CMA_IOC_MAGIC 's'
#define SMOL_CMA_SIZE (32U * 1024U * 1024U)

struct smol_cma_info {
    __u64 dma_addr;
    __u64 size;
    __u32 handle;
    __u32 flags;
};

#define SMOL_CMA_IOC_GET_BUFFER _IOR(SMOL_CMA_IOC_MAGIC, 0x01, struct smol_cma_info)

struct smol_cma_dev {
    struct device *dev;
    struct miscdevice miscdev;
    void *cpu_addr;
    dma_addr_t dma_addr;
    size_t size;
};

static long smol_cma_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct smol_cma_dev *cma = container_of(filp->private_data, struct smol_cma_dev, miscdev);
    struct smol_cma_info info;

    if (cmd != SMOL_CMA_IOC_GET_BUFFER)
        return -ENOTTY;

    memset(&info, 0, sizeof(info));
    info.dma_addr = (u64)cma->dma_addr;
    info.size = (u64)cma->size;

    if (copy_to_user((void __user *)arg, &info, sizeof(info)))
        return -EFAULT;

    return 0;
}

static int smol_cma_mmap(struct file *filp, struct vm_area_struct *vma)
{
    struct smol_cma_dev *cma = container_of(filp->private_data, struct smol_cma_dev, miscdev);
    size_t requested = (size_t)(vma->vm_end - vma->vm_start);

    if (requested > cma->size)
        return -EINVAL;

    return dma_mmap_coherent(cma->dev, vma, cma->cpu_addr, cma->dma_addr, cma->size);
}

static const struct file_operations smol_cma_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = smol_cma_ioctl,
    .compat_ioctl = smol_cma_ioctl,
    .mmap = smol_cma_mmap,
};

static int smol_cma_probe(struct platform_device *pdev)
{
    struct smol_cma_dev *cma;
    u32 size_u32 = SMOL_CMA_SIZE;
    int ret;

    cma = devm_kzalloc(&pdev->dev, sizeof(*cma), GFP_KERNEL);
    if (!cma)
        return -ENOMEM;

    of_property_read_u32(pdev->dev.of_node, "aicas,buffer-size", &size_u32);

    cma->dev = &pdev->dev;
    cma->size = size_u32;
    /*
     * The first SDF1 debug BD maps the helper AXI master through the low DDR
     * aperture. Keep the bring-up buffer below 4 GiB even though the RTL address
     * bus is 40-bit wide.
     */
    dma_set_coherent_mask(&pdev->dev, DMA_BIT_MASK(32));

    cma->cpu_addr = dma_alloc_coherent(&pdev->dev, cma->size, &cma->dma_addr, GFP_KERNEL);
    if (!cma->cpu_addr)
        return -ENOMEM;

    cma->miscdev.minor = MISC_DYNAMIC_MINOR;
    cma->miscdev.name = "smolvlm_llama_fpga_cma";
    cma->miscdev.fops = &smol_cma_fops;
    cma->miscdev.parent = &pdev->dev;

    ret = misc_register(&cma->miscdev);
    if (ret) {
        dma_free_coherent(&pdev->dev, cma->size, cma->cpu_addr, cma->dma_addr);
        return ret;
    }

    platform_set_drvdata(pdev, cma);
    dev_info(&pdev->dev, "allocated coherent buffer dma=%pad size=%zu\n", &cma->dma_addr, cma->size);
    return 0;
}

static int smol_cma_remove(struct platform_device *pdev)
{
    struct smol_cma_dev *cma = platform_get_drvdata(pdev);

    misc_deregister(&cma->miscdev);
    dma_free_coherent(&pdev->dev, cma->size, cma->cpu_addr, cma->dma_addr);
    return 0;
}

static const struct of_device_id smol_cma_of_match[] = {
    { .compatible = "aicas,smolvlm-llama-fpga-cma" },
    { }
};
MODULE_DEVICE_TABLE(of, smol_cma_of_match);

static struct platform_driver smol_cma_driver = {
    .probe = smol_cma_probe,
    .remove = smol_cma_remove,
    .driver = {
        .name = "smolvlm_llama_fpga_cma",
        .of_match_table = smol_cma_of_match,
    },
};
module_platform_driver(smol_cma_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("AICAS2026");
MODULE_DESCRIPTION("SmolVLM llama-fpga coherent CMA helper");
