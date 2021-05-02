# pip3 install opencv-contrib-python

import numpy as np
import copy
import os
import multiprocessing
import logging
import glob
import imageio


import image_processing_fun
from os import listdir
from os.path import isfile, join


np.set_printoptions(threshold=np.inf)
logging.getLogger().setLevel(logging.INFO)


mypath = 'source_images'
outpath = 'results/'


stage = 7


def process_stage1(img_list, parametry):
    image_num = 1
    for image in img_list:
        logging.info("Process ID %s, %s %i:%i", os.getpid(),
                     image, image_num, len(img_list))
        image_num = image_num + 1
        parametry = image_processing_fun.outer_analysis_step(
            mypath,         image, outpath, parametry)


def process_stage2(img_list):
    image_num = 1
    for image in img_list:
        logging.info("Process ID %s, %s %i:%i", os.getpid(),
                     image, image_num, len(img_list))
        image_num = image_num + 1
        image_processing_fun.inner_analysis(
            outpath + '2_outer/',         image, outpath)


def process_stage3(img_list, parametry):
    image_num = 1
    for image in img_list:
        logging.info("Process ID %s, %s %i:%i", os.getpid(),
                     image, image_num, len(img_list))
        image_num = image_num + 1
        parametry = image_processing_fun.bones_analysis_step(
            outpath + '4_inner/', image, outpath, parametry)


def process_stage4(image_folder, model_name):
    img_list = glob.glob(image_folder + '/*.png')
    image_num = 1
    for image in img_list:
        logging.info("Process ID %s, %s %i:%i", os.getpid(),
                     image, image_num, len(img_list))
        image_num = image_num + 1
        image_processing_fun.create_model_3D(image, outpath, model_name)


def thread_function(img_list, parametry):
    image_num = 1
    for image in img_list:
        logging.info("Process ID %s, %s %i:%i", os.getpid(),
                     image, image_num, len(img_list))
        image_num = image_num + 1
        parametry = image_processing_fun.outer_analysis_step(
            mypath,        image, outpath, parametry)


def thread_gif(path, name):
    logging.info("Process ID %s, %s %s", os.getpid(), name, 'Start...')
    onlyfilesgif = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfilesgif.sort()
    images = []
    for filenamegif in onlyfilesgif:
        images.append(imageio.imread(path + '/' + filenamegif))
    imageio.mimsave(path + '.gif', images)
    logging.info("Process ID %s, %s %s", os.getpid(), name, 'Done...')


if __name__ == "__main__":
    formatINFO = "%(asctime)s: %(message)s"
    logging.basicConfig(format=formatINFO,
                        level=logging.INFO, datefmt="%H:%M:%S")

    logging.info("Program start")

    # stworz liste wszystkich plikow
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    # stortuj liste plikow
    onlyfiles.sort()

    # podziel liste na dwie czesci
    index = onlyfiles.index('avf1616b.png')
    list1 = onlyfiles[:index]
    list2 = onlyfiles[index + 1:]
    list1.sort(reverse=True)

    # usu wszsytko z folderu test
    filestoremove = glob.glob(outpath + 'test/*.png')
    for f in filestoremove:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    # etap 1 - skora
    if (stage == 1) | (stage == 0):
        # usun wszsytkie elementy z folderow
        filestoremove = glob.glob(outpath + '0_color/*.png')
        filestoremove.extend(glob.glob(outpath + '1_outer_mask/*.png'))
        filestoremove.extend(glob.glob(outpath + '2_outer/*.png'))

        for f in filestoremove:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

        # analiza srodkowego zdjecia
        logging.info("Stage 1 - skins")
        logging.info("Processing initial image...")
        init_parameters = image_processing_fun.outer_analysis(
            mypath, 'avf1616b.png', outpath)

        # utworzenie dwoch procesow do obrobki pozostalych zdjec
        logging.info("Creating processes...")
        t1 = multiprocessing.Process(
            target=process_stage1, args=(list1, copy.copy(init_parameters)))
        t2 = multiprocessing.Process(
            target=process_stage1, args=(list2, copy.copy(init_parameters)))

        logging.info("Processing...")
        t1.start()
        t2.start()

        t1.join()
        t2.join()

    # etap 2 - tkanka miesniowa
    if (stage == 2) | (stage == 0):
        # usuniecie plikow z folderow
        logging.info("Stage 2 - tissue")
        filestoremove = glob.glob(outpath + '3_inner_mask/*.png')
        filestoremove.extend(glob.glob(outpath + '4_inner/*.png'))

        for f in filestoremove:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

        # tworzenie listy plikow wejsciowych
        file_list = [f for f in listdir(
            outpath + '2_outer/') if isfile(join(outpath + '2_outer/', f))]
        list_len = len(file_list) / 51

        # podzial listy na procesy
        file_list = np.array(file_list)
        file_list = file_list.reshape([int(list_len), -1])

        # tworzenie procesow i ich uruchomienie
        process_list = []
        for i in range(len(file_list)):
            temp_list = file_list[i].tolist()
            p = multiprocessing.Process(
                target=process_stage2, args=[temp_list])
            process_list.append(p)
            p.start()

        logging.info("Processing...")
        for p in process_list:
            p.join()

    # etap 3 - kosci
    if (stage == 3) | (stage == 0):
        logging.info("Stage 3 - bones")

        # tworzenie listy plikow do usuniecia
        filestoremove = glob.glob(outpath + '5_bone1_mask/*.png')
        filestoremove.extend(glob.glob(outpath + '6_bone1/*.png'))
        filestoremove.extend(glob.glob(outpath + '7_bone2_mask/*.png'))
        filestoremove.extend(glob.glob(outpath + '8_bone2/*.png'))

        for f in filestoremove:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

        # tworzenie listy plikow wejsciowych
        onlyfiles = [f for f in listdir(
            outpath + '4_inner/') if isfile(join(outpath + '4_inner/', f))]

        # stortuj liste plikow
        onlyfiles.sort()

        # podziel liste na dwie czesci
        index = onlyfiles.index('avf1616b.png')
        list1 = onlyfiles[:index]
        list2 = onlyfiles[index + 1:]
        list1.sort(reverse=True)

        # analiza obrazu poczatkowego
        init_parameters = image_processing_fun.bones_analysis(
            outpath + '4_inner/', 'avf1616b.png', outpath)

        # tworzenie procesow oraz ich uruchominie
        logging.info("Creating processes...")
        t1 = multiprocessing.Process(
            target=process_stage3, args=(list1, copy.copy(init_parameters)))
        t2 = multiprocessing.Process(
            target=process_stage3, args=(list2, copy.copy(init_parameters)))

        logging.info("Processing...")
        t1.start()
        t2.start()

        t1.join()
        t2.join()

    # etap 4 - vein tracker
    if (stage == 4) | (stage == 0):
        pass

    # etap 5 - tworzenie modelu 3D
    if (stage == 5) | (stage == 0):
        logging.info("Start creating model 3D")

        jobs = []

        model = [[outpath + '1_outer_mask', 'outer'],
                 [outpath + '3_inner_mask', 'inner'],
                 [outpath + '5_bone1_mask', 'bone1'],
                 [outpath + '7_bone2_mask', 'bone2']]

        for i in model:
            p = multiprocessing.Process(
                target=process_stage4, args=(i[0], i[1]))
            jobs.append(p)
            p.start()

        logging.info("Processing...")
        for p in jobs:
            p.join()

    # etap 6 - prostowanie modelu 3D
    if (stage == 6) | (stage == 0):

        logging.info("Start creating model 3D - straightening")
        image_processing_fun.model_straightening(outpath)

    # etap 8 - tworzenie animacji GIF
    if (stage == 7) | (stage == 0):

        logging.info("Start create GIF")

        jobs = []

        gif = [[outpath + '/0_color', 'color.gif'],
               [outpath + '/1_outer_mask', 'outer_mask.gif'],
               [outpath + '/2_outer', 'outer.gif'],
               [outpath + '/3_inner_mask', 'inner_mask.gif'],
               [outpath + '/4_inner', 'inner.gif'],
               [outpath + '/5_bone1_mask', 'bone1_mask.gif'],
               [outpath + '/6_bone1', 'bone1.gif'],
               [outpath + '/7_bone2_mask', 'bone2_mask.gif'],
               [outpath + '/8_bone2', 'bone2.gif']]  # ,

        for i in gif:
            p = multiprocessing.Process(target=thread_gif, args=(i[0], i[1]))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        logging.info("Program end")
