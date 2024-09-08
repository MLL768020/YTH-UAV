# 计算长和宽的差值
dim_diff = np.abs(high - width)

# 计算上下左右分别需要填充多少个维度
pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
pad = (0, 0, pad1, pad2) if high <= width else (pad1, pad2, 0, 0)
top, bottom, left, right = pad

# 使用opencv的copyMakeBorder函数进行填充
pad_value = 0
img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, pad_value)
print(img_dst.shape)
