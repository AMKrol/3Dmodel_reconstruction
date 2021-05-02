import numpy as np
import sys
import logging
import pandas as pd
import math

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    formatINFO = "%(asctime)s: %(message)s"
    logging.basicConfig(format=formatINFO, level=logging.INFO, datefmt="%H:%M:%S")
    outpath = 'test/'
    outname = 'vein2'
    file1 = open(outpath + 'modelV3/' + outname + '.xyz', 'r')

    Lines = file1.readlines()
    file1.close()

    pointlist = []
    for line in Lines:
        test = line.strip()
        test = test.split(' ')
        pointlist.append(test)
    
    df = pd.DataFrame(pointlist, columns=['X', 'Y', 'Z'], dtype = float)
    df = df.dropna()
    df = df.astype('float')
    df['nx'] = ''
    df['ny'] = ''
    df['nz'] = ''

    uniqueZ = df['Z'].unique().tolist()
    vortexnum = len(df.index)

    print(df)
    for Zstep in uniqueZ:
        df_Z = df[df.Z == Zstep]

        columnX = df_Z['X']
        columnY = df_Z['Y']

        maxX = columnX.max()
        minX = columnX.min()

        maxY = columnY.max()
        minY = columnY.min()

        meanX = (maxX + minX)/2
        meanY = (maxY + minY)/2

        df.loc[df['Z'] == Zstep, 'nz'] = 0
        df.loc[df['Z'] == Zstep, 'nx'] = np.cos(np.arctan((df['Y'] - meanY)/(df['X'] - meanX)))*np.sign(df['X'] - meanX)
        df.loc[df['Z'] == Zstep, 'ny'] = np.sin(np.arctan((df['Y'] - meanY)/(df['X'] - meanX)))*np.sign(df['X'] - meanX)
        
    df = df.astype('float')

    print(df)
    
    headerstr = 'ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex ' + str(vortexnum) + '\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n'

    np.savetxt('test/modelV3/' + outname + '.ply', df.values, fmt='%10.6f', delimiter=' ', header=headerstr, comments = '')
