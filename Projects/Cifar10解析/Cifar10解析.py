import pickle
import PIL.Image as image
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data():
    data_batch1 = unpickle('./data/cifar-10-batches-py/data_batch_1')
    data_batch2 = unpickle('./data/cifar-10-batches-py/data_batch_2')
    data_batch3 = unpickle('./data/cifar-10-batches-py/data_batch_3')
    data_batch4 = unpickle('./data/cifar-10-batches-py/data_batch_4')
    data_batch5 = unpickle('./data/cifar-10-batches-py/data_batch_5')

    '''
    字符串前边的字母b,r,u的含义：
    
    b'Hello,world!'
    python3.x里默认的str是(py2.x里的)unicode, bytes是(py2.x)的str, b”“前缀代表的就是bytes ；
    python2.x里, b前缀没什么具体意义， 只是为了兼容python3.x的这种写法。

    r'\s\d{3,6}'
    常用于正则表达式或文件绝对地址等，该字母后面一般一般接转义字符，有特殊含义的字符。所以，要使用转义字符，通常字符串前面要加r。

    u'生日快乐'
    u后面的字符串表示使用Unicode编码，因为中文也有对应的Unicode编码，所以常用于中文字符串的前面，防止出现乱码。
    '''

    img_batch1 = data_batch1[b'data']
    img_batch2 = data_batch2[b'data']
    img_batch3 = data_batch3[b'data']
    img_batch4 = data_batch4[b'data']
    img_batch5 = data_batch5[b'data']

    return img_batch1, img_batch2, img_batch3, img_batch4, img_batch5


def reconstruct_data(img):
    # 得到第一张图像
    img_0 = img[0]
    img_reshape = img_0.reshape(3, 32, 32)

    # 构建RGB三通道
    r = image.fromarray(img_reshape[0]).convert('L')
    g = image.fromarray(img_reshape[1]).convert('L')
    b = image.fromarray(img_reshape[2]).convert('L')

    # 融合三通道
    img_m = image.merge('RGB', (r, g, b))

    # 画图
    plt.imshow(img_m)
    plt.show()


if __name__ == '__main__':
    img_batch1, _, _, _, _ = get_data()
    print(img_batch1.shape)  # 显示为（10000，3072）, 3072 = 3 * 32 * 32

    reconstruct_data(img_batch1)

