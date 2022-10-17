from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(r"E:\code\python_code\Pytorch_gogo\cs231n-2019-assignments-master\Pytorch_learning_小土堆\logs")


# writer.add_image()
for i in range(100):
    writer.add_scalar("y=x", i, i)     # add_scalar 添加标量

writer.close()