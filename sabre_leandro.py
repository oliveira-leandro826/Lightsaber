import pylab
import imageio
import cv2
import matplotlib.pyplot as plt
from videoreader import VideoReader
import cv2
import skimage as sk
import numpy as np
from skimage.morphology import disk, ball
import imageio

## Função utilizada da somar os valores de pixels na imagem resultante da extração do canal de cor
## e a imagem segmentada por meio da limiarização.

def imadd(imagem_original, segmentacao):
    # Obtém os valores correspondentes a dimensão da imagem.
    width, height = imagem_original.shape
    # Cria uma nova imagem com as mesmas dimensões que irá receber os valores somados.
    nova_imagem = np.zeros((width, height))
    # Iteração entre as dimensões.
    for w in range(width):
        for h in range(height):
            # Se a soma dos valores de pixels das duas imagens for superior a 255, ele irá truncar o valor em 255.
            if (segmentacao[w,h] + imagem_original[w, h]) > 255:
                nova_imagem[w,h] = 255
            else:
            # Caso o valor não seja superior, realiza a soma entre os valores, uma vez que a segmentação só terá
            # 0 e 255.
                nova_imagem[w,h] = segmentacao[w,h] + imagem_original[w, h]

    # Retorna a imagem
    return nova_imagem

## Essa função irá processar o frame atual e irá aplicar o efeito do sabre de luz.
def processing_frame(frame):

    ## Seleciona os canais de cores do sistema de cor RGB
    red_channel = frame[:,:,0]
    green_channel = frame[:,:,1]
    blue_channel = frame[:,:,2]

    # Realça os valores do canal verde (utilizado como cor objetivo no vídeo)
    realce = green_channel - blue_channel/2 - red_channel/2

    # Aplica a limiarização na imagem realçada, resultando em uma imagem binária
    segmentada = realce > 10

    # Aplica a erosão nas bordas da imagem segmentada para reduzir a largura do sabre de luz
    footprint = disk(1)
    eroded = sk.morphology.erosion(segmentada, footprint)

    # Aplica a suavização nas bordas da imagem para eliminar o serrilhado. Além disso, irá suavizar
    # mais o canal verde, dando o efeito de sabre com  bordas verdes.
    filter_gau_green = sk.filters.gaussian(segmentada, sigma=3.0)
    filter_gau_red_blue = sk.filters.gaussian(segmentada, sigma=1.0)

    # Normalização dos valores entre 0 e 255 para futura adição das imagens
    filter_gau_red_blue *= 255/filter_gau_red_blue.max()
    filter_gau_green *= 255/filter_gau_green.max()

    # Adição dos canais de cores com as respectivas imagens segmentadas e suavizadas.
    green_final = imadd(filter_gau_green, green_channel)
    blue_final = imadd(filter_gau_red_blue, blue_channel)
    red_final = imadd(filter_gau_red_blue, red_channel)

    # Concatenação entre os canais de cores.
    final_frame = np.dstack((red_final.astype(int), green_final.astype(int), blue_final.astype(int)))

    # Retorna o frame processado e o valor máximo da segmentação.
    return final_frame, filter_gau_red_blue.max()

# Leitura e escrita do arquivo.
filename = '/root/codigos_luis/model/sabre_leandro_2.mp4'
reader = imageio.get_reader(filename)
writer = imageio.get_writer('cockatoo_gray.mp4', fps=24)

# Iteração entre os frames.
for i, im in enumerate(reader):
    # Processa o frame atual para aplicar o efeito do sabre de luz.
    saida, maximo = processing_frame(im)

    # Verifica se a imagem foi segmentada.
    if maximo > 240:
        print('Máximo: ', maximo)
        im_processada = saida
        writer.append_data(im_processada)
    else:
        writer.append_data(im)

# Salva o vídeo
writer.close()
