# Yolov8-Hisi

    yolov8在hisi3536a推理，./build.sh 编译即可。

## yolov8源码需要修改部分

### DFL模块修改

```python
class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

```

更换为如下：

```python
class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, h, w = x.shape  # batch, channels, anchors
        return self.conv(x.permute(0,2,3,1).view(b, h*w, 4, c//4).softmax(3).permute(0,3,1,2)).view(b, h*w, 4)
```

### Detect的forward修改为如下：

```python
def forward(self, x):
        shape = x[0].shape  # BCHW

        if self.export == True:             
            for i in range(self.nl):
                boxs = self.dfl(self.cv2[i](x[i]))
                print(boxs.shape)
                cls = self.cv3[i](x[i]).permute(0,2,3,1)
                b,h,w,c = cls.shape
                # print(b,h,w,c)
                cls = cls.view(b, h*w, c)
                print(cls.shape)
                x[i] = torch.cat(((boxs, cls.sigmoid())), 2)
            return x
        else:

            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            if self.training:
                return x
            elif self.dynamic or self.shape != shape:
                self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
                self.shape = shape

            x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
            if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
                box = x_cat[:, :self.reg_max * 4]
                cls = x_cat[:, self.reg_max * 4:]
            else:
                box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            y = torch.cat((dbox, cls.sigmoid()), 1)
            return y if self.export else (y, x)
```

## 导出模型

```python
    path = "/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/GWM/weights/best.pt"
    yolo = YOLO(path)
    success = yolo.export(format="onnx",simplify = True,imgsz=[384,640])
    导出模型后在海思工具上转换om模型即可在本demo代码上使用。
```
