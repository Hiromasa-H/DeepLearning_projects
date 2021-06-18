import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import argparse


parser = argparse.ArgumentParser(description='Generate a stylized image.')
parser.add_argument('--c',
                    help='path to content image')
parser.add_argument('--s',
                    help='path to style image')
parser.add_argument('--g',
                    help='desired path to generated image')
parser.add_argument('--ih',
                    help='image height',default=356)
parser.add_argument('--iw',
                    help='image width',default=356)
parser.add_argument('--ne',
                    help='number of epochs (default=6000',default=6000)
parser.add_argument('--lr', 
                    help='learning rate (default=0.001)',default=0.001)
parser.add_argument('--alpha', 
                    help='alpha (default=1)',default=1)
parser.add_argument('--beta', 
                    help='beta (default=0.01)',default=0.01)
args = parser.parse_args()



vgg19 = models.vgg19(pretrained=True).features

#print(vgg19)
# layers = ['0','5','10','19','28']

class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()

    self.chosen_layers = ['0','5','10','19','28']
    self.model = models.vgg19(pretrained=True).features[:29]
  def forward(self,x):
    features = []

    for index, layer in enumerate(self.model):
      x = layer(x)

      if str(index) in self.chosen_layers:
        features.append(x)

    return features


def load_image(image_name):
  image = Image.open(image_name)
  image = loader(image).unsqueeze(0) #add additional dimension for batch size (which in this case should be one)
  return image.to(device)

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

image_height = args.ih
image_width = args.iw

loader = transforms.Compose(
  [
    transforms.Resize((image_height,image_width)), #reformat the image size to make computing the loss possible
    transforms.ToTensor(),
  ])


content_image = load_image(str(args.c))
style_image = load_image(str(args.s))

model = VGG().to(device).eval()

# generated_image = torch.random(content_image.shape,device=device, require_grad=True)
generated_image = content_image.clone().requires_grad_(True)

#HyperParams

num_epochs = int(args.ne)
lr = float(args.lr)
alpha = float(args.alpha)
beta = float(args.beta)
optimizer = optim.Adam([generated_image],lr=lr)

for epoch in range(num_epochs):
  generated_features = model(generated_image)
  content_features = model(content_image)
  style_features = model(style_image)

  style_loss = content_loss = 0

  for gf,cf,sf in zip(generated_features,content_features,style_features):
    #content loss

    batch_size, channel, height, width = gf.shape
    content_loss += torch.mean((gf-cf)**2)

    #style loss

    Gram_gen = gf.view(channel,height*width).mm(
      gf.view(channel,height*width).t())

    Gram_style = sf.view(channel,height*width).mm(
      sf.view(channel,height*width).t())

    style_loss += torch.mean((Gram_gen - Gram_style)**2)

  total_loss = alpha * content_loss + beta * style_loss
  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()
  
  # if epoch % 1000==0:
  #   print(f"epoch:{epoch}, loss:{total_loss}")
  #   save_image(generated_image,str(args.generated))

save_image(generated_image,str(args.g))