#!/usr/bin/env python
#
# author Mingxu Hu
# author Hongkun Yu
# author Shouqing Li
#
# version 1.4.11.081101
# copyright THUNDER Non-Commercial Software License Agreement
#
# ChangeLog
# AUTHOR      | TIME       | VERSION       | DESCRIPTION
# ------      | ----       | -------       | -----------
# Mingxu   Hu | 2015/03/23 | 0.0.1.050323  | new file
# Hongkun  Yu | 2015/03/23 | 0.0.1.050323  | new file
# Shouqing Li | 2018/11/01 | 1.4.11.081101 | add option
#
# STAR_2_THU.py can translate a STAR file from RELION into a THU file used for THUNDER.

import math
import os,sys
import re
from optparse import OptionParser

def euler_to_quaternion(src):

    psi = math.radians(src[0])
    theta = math.radians(src[1])
    phi = math.radians(src[2])

    w = math.cos((phi + psi) / 2) * math.cos(theta / 2)
    x = math.sin((phi - psi) / 2) * math.sin(theta / 2)
    y = math.cos((phi - psi) / 2) * math.sin(theta / 2)
    z = math.sin((phi + psi) / 2) * math.cos(theta / 2)

    return w, x, y, z

def main():

    prog_name = os.path.basename(sys.argv[0])
    usage = """
    Transform STAR file to THU file.
    {prog} < -i input_star file> < -o output_thu file >
    """.format(prog = prog_name)

    optParser = OptionParser(usage)
    optParser.add_option("-i", \
                         "--input", \
                         action = "store", \
                         type = "string", \
                         dest = "input_star", \
                         help = "Input RELION data.star file.")
    optParser.add_option("-o", \
                         "--output", \
                         action = "store", \
                         type = "string", \
                         dest = "output_thu", \
                         help = "Output THUNDER data.thu file.")
    (options, args) = optParser.parse_args()

    header_dict = {}
    flag = 0

    if len(sys.argv) == 1:
        print usage
        print "    For more detail, see '-h' or '--help'."

    if options.output_thu:
        fout = open(options.output_thu, "w")

        try:
            fin = open(options.input_star, "r")
        except:
            print "Please input a proper thu file."
            exit()

        for line in fin.readlines():

            if (line[0] == '#'):
                flag = flag + 1
                continue
            else:
                continue

        fin.close()
        fin = open(options.input_star, "r")

        if flag == 1 or flag == 0:
            for num, line in enumerate(fin):

                if (line[0] == '#'):
                    continue

                sline = line.strip()

                if not sline or (' ' not in sline):
                    continue

                match = re.match(r'_rln(\w+)\s+#(\d+)', sline)

                if match:
                    header_dict[match.group(1).lower()] = int(match.group(2)) - 1
                    continue

                sp = sline.split()

                try:
                    ps = float(sp[header_dict['phaseshift']]) * (math.pi / 180)

                except (ValueError, IndexError):
                    sys.stderr.write('Warning: skipping line #{} ({}) that cannot be parsed\n'.format(num + 1, sline))
                    continue

                except KeyError:
                    ps = 0

                try:
                    gI = int(sp[header_dict['groupnumber']])

                except (ValueError, IndexError):
                    sys.stderr.write('Warning: skipping line #{} ({}) that cannot be parsed\n'.format(num + 1, sline))
                    continue

                except KeyError:
                    gI = 1

                try:
                    volt = float(sp[header_dict['voltage']]) * 1000.0
                    dU = float(sp[header_dict['defocusu']])
                    dV = float(sp[header_dict['defocusv']])
                    dT = float(sp[header_dict['defocusangle']]) * (math.pi / 180)
                    c = float(sp[header_dict['sphericalaberration']]) * 1e7
                    aC = float(sp[header_dict['amplitudecontrast']])
                    iN = sp[header_dict['imagename']]
                    mN = sp[header_dict['micrographname']]
                    coordX = float(sp[header_dict['coordinatex']])
                    coordY = float(sp[header_dict['coordinatey']])

                except (ValueError, IndexError):
                    sys.stderr.write('Warning: skipping line #{} ({}) that cannot be parsed\n'.format(num + 1, sline))
                    continue

                except KeyError as e:
                    sys.stderr.write('Header does not include {}. This is an invalid star file\n'.format(str(e)))
                    sys.exit(2)

                try:
                    phi = float(sp[header_dict['anglerot']])
                    theta = float(sp[header_dict['angletilt']])
                    psi = float(sp[header_dict['anglepsi']])

                except (ValueError, IndexError):
                    sys.stderr.write('Warning: skipping line #{} ({}) that cannot be parsed\n'.format(num + 1, sline))
                    continue

                except KeyError:
                    phi = 0
                    theta = 0
                    psi = 0

                quat0, quat1, quat2, quat3 = euler_to_quaternion([phi, theta, psi])

                fout.write('{volt:18.6f} \
                            {dU:18.6f} \
                            {dV:18.6f} \
                            {dT:18.6f} \
                            {c:18.6f} \
                            {aC:18.6f} \
                            {phaseShift:18.6f} \
                            {iN} \
                            {mN} \
                            {coordX:18.6f} \
                            {coordY:18.6f} \
                            {gI} \
                            {classI} \
                            {quat0:18.6f} \
                            {quat1:18.6f} \
                            {quat2:18.6f} \
                            {quat3:18.6f} \
                            {stdRot0:18.6f} \
                            {stdRot1:18.6f} \
                            {stdRot2:18.6f} \
                            {transX:18.6f} \
                            {transY:18.6f} \
                            {stdTransX:18.6f} \
                            {stdTransY:18.6f} \
                            {df:18.6f} \
                            {stdDf:18.6f} \
                            {score:18.6f} \n'.format(volt = volt,
                                              dU = dU,
                                              dV = dV,
                                              dT = dT,
                                              c = c,
                                              aC = aC,
                                              phaseShift = ps,
                                              iN = iN,
                                              mN = mN,
                                              coordX = coordX,
                                              coordY = coordY,
                                              gI = gI,
                                              classI = 0,
                                              quat0 = quat0,
                                              quat1 = quat1,
                                              quat2 = quat2,
                                              quat3 = quat3,
                                              stdRot0 = 0,
                                              stdRot1 = 0,
                                              stdRot2 = 0,
                                              transX = 0,
                                              transY = 0,
                                              stdTransX = 0,
                                              stdTransY = 0,
                                              df = 1,
                                              stdDf = 0,
                                              score = 0))

        if flag == 2:

            part = 0
            door = 1
            dic = []

            for num, line in enumerate(fin):
                if (line[0] == '#'):
                    part = part + 1
                    continue

                if (door == 1) and (part == 2):
                    optics_dict = header_dict
                    header_dict = {}
                    door = 0

                sline = line.strip()

                if not sline or (' ' not in sline):
                    continue

                match = re.match(r'_rln(\w+)\s+#(\d+)', sline)

                if match:
                    header_dict[match.group(1).lower()] = int(match.group(2)) - 1
                    continue

                sp = sline.split()

                if part == 1:

                    dic.append(sp)

                if part == 2:

                    try:
                        gp = int(sp[header_dict['opticsgroup']])

                    except (ValueError, IndexError):
                        sys.stderr.write('Warning: skipping line #{} ({}) that cannot be parsed\n'.format(num + 1, sline))
                        continue

                    except KeyError as e:
                        sys.stderr.write('Header does not include {}. This is an invalid star file\n'.format(str(e)))
                        sys.exit(2)

                    else:
                        volt = float(dic[gp - 1][optics_dict['voltage']]) * 1000.0
                        c = float(dic[gp - 1][optics_dict['sphericalaberration']]) * 1e7
                        aC = float(dic[gp - 1][optics_dict['amplitudecontrast']])

                    try:
                        ps = float(sp[header_dict['phaseshift']]) * (math.pi / 180)

                    except (ValueError, IndexError):
                        sys.stderr.write('Warning: skipping line #{} ({}) that cannot be parsed\n'.format(num + 1, sline))
                        continue

                    except KeyError:
                        ps = 0

                    try:
                        gI = int(sp[header_dict['groupnumber']])

                    except (ValueError, IndexError):
                        sys.stderr.write('Warning: skipping line #{} ({}) that cannot be parsed\n'.format(num + 1, sline))
                        continue

                    except KeyError:
                        gI = 1

                    try:
                        phi = float(sp[header_dict['anglerot']])
                        theta = float(sp[header_dict['angletilt']])
                        psi = float(sp[header_dict['anglepsi']])

                    except (ValueError, IndexError):
                        sys.stderr.write('Warning: skipping line #{} ({}) that cannot be parsed\n'.format(num + 1, sline))
                        continue

                    except KeyError:
                        phi = 0
                        theta = 0
                        psi = 0

                    try:
                        dU = float(sp[header_dict['defocusu']])
                        dV = float(sp[header_dict['defocusv']])
                        dT = float(sp[header_dict['defocusangle']]) * (math.pi / 180)
                        iN = sp[header_dict['imagename']]
                        mN = sp[header_dict['micrographname']]
                        coordX = float(sp[header_dict['coordinatex']])
                        coordY = float(sp[header_dict['coordinatey']])

                    except (ValueError, IndexError):
                        sys.stderr.write('Warning: skipping line #{} ({}) that cannot be parsed\n'.format(num + 1, sline))
                        continue

                    except KeyError as e:
                        sys.stderr.write('Header does not include {}. This is an invalid star file\n'.format(str(e)))
                        sys.exit(2)

                    quat0, quat1, quat2, quat3 = euler_to_quaternion([phi, theta, psi])

                    fout.write('{volt:18.6f} \
                                {dU:18.6f} \
                                {dV:18.6f} \
                                {dT:18.6f} \
                                {c:18.6f} \
                                {aC:18.6f} \
                                {phaseShift:18.6f} \
                                {iN} \
                                {mN} \
                                {coordX:18.6f} \
                                {coordY:18.6f} \
                                {gI} \
                                {classI} \
                                {quat0:18.6f} \
                                {quat1:18.6f} \
                                {quat2:18.6f} \
                                {quat3:18.6f} \
                                {stdRot0:18.6f} \
                                {stdRot1:18.6f} \
                                {stdRot2:18.6f} \
                                {transX:18.6f} \
                                {transY:18.6f} \
                                {stdTransX:18.6f} \
                                {stdTransY:18.6f} \
                                {df:18.6f} \
                                {stdDf:18.6f} \
                                {score:18.6f} \n'.format(volt = volt,
                                                  dU = dU,
                                                  dV = dV,
                                                  dT = dT,
                                                  c = c,
                                                  aC = aC,
                                                  phaseShift = ps,
                                                  iN = iN,
                                                  mN = mN,
                                                  coordX = coordX,
                                                  coordY = coordY,
                                                  gI = gI,
                                                  classI = 0,
                                                  quat0 = quat0,
                                                  quat1 = quat1,
                                                  quat2 = quat2,
                                                  quat3 = quat3,
                                                  stdRot0 = 0,
                                                  stdRot1 = 0,
                                                  stdRot2 = 0,
                                                  transX = 0,
                                                  transY = 0,
                                                  stdTransX = 0,
                                                  stdTransY = 0,
                                                  df = 1,
                                                  stdDf = 0,
                                                  score = 0))

        fin.close()
        fout.close()

if __name__ == "__main__":
    main()
