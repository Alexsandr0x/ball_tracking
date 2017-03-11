# coding=utf-8

import cv2
import json
import numpy as np

file_directory = "config.json"
try:
    config_ = json.loads(open(file_directory).read())
except Exception as error:
    raise error


def find_ball(image_path, lower_color, upper_color, out_file='frame'):
    """
    Metodo que implementa todo o processo de encontrar a bola na imagem.
    Abre uma janela com a imagem de image_path com o ponto e retorna coordenadas da
    bola encontrada.

    :param image_path: o caminho da imagem a ser processada
    :param lower_color: limite inferior para cor que irá procurar em nparray(B, G, R)
    :param upper_color: limite superior para cor que irá procurar em nparray(B, G, R)
    :param out_file: arquivo de saida da imagem.
    :return (coords, image): coordenadas (x, y) da bola encontrada e imagem aṕos processamento
    """
    center = None

    picture = cv2.imread(image_path)

    picture_no_noise = cv2.blur(picture, (5, 5))
    # Podem haver casos onde a imagem estaja com muito ruido, nesse caso podemos usar
    # ou um algoritmo simples de blur ou um algoritmo de "denoise" mais sofisticado
    # nesse caso acabei usando o blur simples por ser mais rapido e ser o suficiente
    # para proximo do centro da bola as predições.

    # picture_no_noise = cv2.fastNlMeansDenoisingColored(picture, None, 15, 10, 7, 21)
    # picture_no_noise = picture

    picture_hsv = cv2.cvtColor(picture_no_noise, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(picture_hsv, lower_color, upper_color)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        m = cv2.moments(c)

        if m["m00"] != 0:
            center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
            cv2.circle(picture_no_noise, center, 10, (0, 255, 0), -1)

    image_file = 'out/{}.png'.format(out_file)
    cv2.imwrite(image_file, picture_no_noise)

    return center, picture_no_noise

colors = config_['colors']['ball_jpg']
l_color = np.array(colors[0])
u_color = np.array(colors[1])

paths = config_["image_path"]

if isinstance(paths, list):
    for i, path in enumerate(paths):
        print find_ball(path, l_color, u_color, 'frame_{}'.format(i))[0]
elif isinstance(paths, basestring):
    print find_ball(paths, l_color, u_color)[0]
else:
    print '[ERROR]: \'image_path\' in config file must be string or list of strings!'
    exit()
