444#!/usr/bin/env python3
# -*- coding: utf-8 -*-"""
###############################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift
np.set_printoptions(threshold=np.inf)


"""
グラフやフォント等の設定tesutodayo
"""
plt.style.use('seaborn-bright')
plt.rcParams['font.family'] = 'Times New Roman'  # 全体のフォント
#plt.rcParams['font.family'] = 'IPAexGothic'  # 全体のフォントを設定 (日本語表示の場合に追加)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.default'] = 'it'   # 数式で通常使用するフォント．

""" Macbook ProでのPython 3.7系用に微調整 (EMC Europe 2019原稿から)"""
plt.rcParams['xtick.direction'] = 'inout'
plt.rcParams['ytick.direction'] = 'inout'
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 2.0
plt.rcParams["legend.markerscale"] = 1.5
plt.rcParams['xtick.major.pad'] = 11
plt.rcParams['ytick.major.pad'] = 11
"""これは一つの分散型遅延線路(三つの線路を並列)のプログラムです"""
###############################################################
"""メイン関数"""
if __name__ == '__main__':
    #パラメータｃｓｖファイルの読み込み

    pd.set_option("display.max_rows",10)
    df_csv = pd.read_csv("Book1.csv",encoding = "shift-jis",index_col =0)
    #list_df = pd.DataFrame( columns=["S21mag1","S21mag2","S21mag3","S21NGD1","S21NGD2","S21NGD3","Cenfre1","Cenfre2","Cenfre3"])
    #list_df = pd.DataFrame( columns=["0","1","0","0","0","0","0","0","0","0","0","0","0","0","0",])
    list_df = pd.DataFrame( columns=["freqwid1","freqwid2","freqwid3"])
    #"freqwid1","freqwid2","freqwid3"
    print(len(df_csv))
    for i in range(len(df_csv)) :
        t1=df_csv.values[i,0]
        t2=df_csv.values[i,1]
        t3=df_csv.values[i,2]
        print(t1,t2,t3)
        #t1,t2,t3 = 0.26,0.31,0.36
        # 解析周波数範囲
        delta_f = 0.0001e9
        fmin = 0.1e9
        fmax = 3.2e9

        # 配列の初期化 [周波数軸]
        freq = np.arange(fmin, fmax, delta_f)
        freq2 = freq[0:(len(freq) - 1)] + np.diff(freq) / 2.0

        # 配列の初期化 [S21, A]
        freqS21mag = np.zeros([len(freq), 2])
        freqS21phase = np.zeros([len(freq), 2])
        freqS21phase2 = np.zeros([len(freq), 2])
        freqHRmag = np.zeros([len(freq), 2])
        freqHRphase = np.zeros([len(freq), 2])
        freqHRphase2 = np.zeros([len(freq), 2])

        freqS21mag[:, 0] = freq
        freqS21phase[:, 0] = freq
        freqS21phase2[:, 0] = freq
        freqHRmag[:, 0] = freq
        freqHRphase[:, 0] = freq
        freqHRphase2[:, 0] = freq

        # 伝送線路全体での損失 [in dB]
        alpha_tgt = 0.3

        # 伝送線路条件
        # LC: p.578の条件から計算
        # Lengs: 論文中の物理長では共振点がズレたので，伝搬速度と遅延時間から逆算
        # 抵抗値: 損失をシリーズ抵抗のみで表現
        """ 線路 1 """
        Cs1 = 7.79498e-11
        Ls1 = 4.00713e-7
        Z1 = np.sqrt(Ls1 / Cs1)
        vp1 = 1.0 / np.sqrt(Ls1 * Cs1)
        # Lengs1 = 41.5e-3
        Lengs1 = vp1 * t1*1e-9
        Rs1 = alpha_tgt * 2 * Z1 / (8.686 * Lengs1)

        """ 線路 2 """
        Cs2 = 7.79498e-11
        Ls2 = 4.00713e-7
        Z2 = np.sqrt(Ls2 / Cs2)
        vp2 = 1.0 / np.sqrt(Ls2 * Cs2)
        # Lengs2 = 47.5e-3
        Lengs2 = vp2 * t2*1e-9
        Rs2 = alpha_tgt * 2 * Z2 / (8.686 * Lengs2)
        """線路３"""
        Cs3 = 7.79498e-11
        Ls3 = 4.00713e-7
        Z3 = np.sqrt(Ls3 / Cs3)
        vp3 = 1.0 / np.sqrt(Ls3 * Cs3)

        Lengs3 = vp3 * t3*1e-9
        Rs3 = alpha_tgt * 2 * Z3 / (8.686 * Lengs3)
        """線路4"""
        Cs4 = 7.79498e-11
        Ls4 = 4.00713e-7
        Z4 = np.sqrt(Ls4 / Cs4)
        vp4 = 1.0 / np.sqrt(Ls4 * Cs4)

        Lengs4 = vp4 * 0.41e-9
        Rs4 = alpha_tgt * 2 * Z4 / (8.686 * Lengs4)
        """
        CsG = 7.79498e-11
        LsG = 4.00713e-7
        ZG = np.sqrt(LsG / CsG)
        vpG = 1.0 / np.sqrt(LsG * CsG)

        LengsG = vpG * 0.e-9
        RsG = alpha_tgt * 2 * ZG / (8.686 * LengsG)
        """
        """線路5"""
        Cs5 = 7.79498e-11
        Ls5 = 4.00713e-7
        Z5 = np.sqrt(Ls5 / Cs5)
        vp5 = 1.0 / np.sqrt(Ls5 * Cs5)

        Lengs5 = vp5 * 0.46e-9
        Rs5 = alpha_tgt * 2 * Z5 / (8.686 * Lengs5)
        """線路6"""
        Cs6 = 7.79498e-11
        Ls6 = 4.00713e-7
        Z6 = np.sqrt(Ls6 / Cs6)
        vp6 = 1.0 / np.sqrt(Ls6 * Cs6)

        Lengs6 = vp6 * 0.51e-9
        Rs6 = alpha_tgt * 2 * Z6 / (8.686 * Lengs6)




        #print(Rs2)
        #print("t1="+str(Lengs1/vp1),"t2="+str(Lengs2/vp2),"t3="+str(Lengs3/vp3))
        # 周波数特性の算出
        for n in range(0, len(freq)):

            # 線路1のF行列 => Y行列
            gamma1 = Rs1 / (2.0 * Z1) + 2.0j * np.pi * freq[n] * np.sqrt(Ls1 * Cs1)
            Z1 = np.sqrt(Ls1 / Cs1)
            F1 = np.array([[np.cosh(gamma1 * Lengs1), 1.0 * Z1 * np.sinh(gamma1 * Lengs1)], [1.0 / Z1 * np.sinh(gamma1 * Lengs1), np.cosh(gamma1 * Lengs1)]])
            A1, B1, C1, D1 = F1[0, 0], F1[0, 1], F1[1, 0], F1[1, 1]
            Y1 = np.array([[D1 / B1, -(A1 * D1 - B1 * C1) / B1], [-1.0 / B1, A1 / B1]])

            # 線路2のF行列 => Y行列
            gamma2 = Rs2 / (2.0 * Z2) + 2.0j * np.pi * freq[n] * np.sqrt(Ls2 * Cs2)
            Z2 = np.sqrt(Ls2 / Cs2)
            F2 = np.array([[np.cosh(gamma2 * Lengs2), 1.0 * Z2 * np.sinh(gamma2 * Lengs2)], [1.0 / Z2 * np.sinh(gamma2 * Lengs2), np.cosh(gamma2 * Lengs2)]])
            A2, B2, C2, D2 = F2[0, 0], F2[0, 1], F2[1, 0], F2[1, 1]
            Y2 = np.array([[D2 / B2, -(A2 * D2 - B2 * C2) / B2], [-1.0 / B2, A2 / B2]])

            # 線路3のF行列 => Y行列
            gamma3 = Rs3 / (2.0 * Z3) + 2.0j * np.pi * freq[n] * np.sqrt(Ls3 * Cs3)
            Z3 = np.sqrt(Ls3 / Cs3)
            F3 = np.array([[np.cosh(gamma3 * Lengs3), 1.0 * Z3 * np.sinh(gamma3 * Lengs3)], [1.0 / Z3 * np.sinh(gamma3 * Lengs3), np.cosh(gamma3 * Lengs3)]])
            A3, B3, C3, D3 = F3[0, 0], F3[0, 1], F3[1, 0], F3[1, 1]
            Y3 = np.array([[D3 / B3, -(A3 * D3 - B3 * C3) / B3], [-1.0 / B3, A3 / B3]])

            # 線路4のF行列 => Y行列
            gamma4 = Rs4 / (2.0 * Z4) + 2.0j * np.pi * freq[n] * np.sqrt(Ls4 * Cs4)
            Z4 = np.sqrt(Ls4 / Cs4)
            F4 = np.array([[np.cosh(gamma4 * Lengs4), 1.0 * Z4 * np.sinh(gamma4 * Lengs4)], [1.0 / Z4 * np.sinh(gamma4 * Lengs4), np.cosh(gamma4 * Lengs4)]])
            A4, B4, C4, D4 = F4[0, 0], F4[0, 1], F4[1, 0], F4[1, 1]
            Y4 = np.array([[D4 / B4, -(A4 * D4 - B4 * C4) / B4], [-1.0 / B4, A4 / B4]])
            # 線路5のF行列 => Y行列
            gamma5 = Rs5 / (2.0 * Z5) + 2.0j * np.pi * freq[n] * np.sqrt(Ls5 * Cs5)
            Z5 = np.sqrt(Ls5 / Cs5)
            F5 = np.array([[np.cosh(gamma5 * Lengs5), 1.0 * Z5 * np.sinh(gamma5 * Lengs5)], [1.0 / Z5 * np.sinh(gamma5 * Lengs5), np.cosh(gamma5 * Lengs5)]])
            A5, B5, C5, D5 = F5[0, 0], F5[0, 1], F5[1, 0], F5[1, 1]
            Y5 = np.array([[D5 / B5, -(A5 * D5 - B5 * C5) / B5], [-1.0 / B5, A5 / B5]])

            # 線路6のF行列 => Y行列
            gamma6 = Rs6 / (2.0 * Z6) + 2.0j * np.pi * freq[n] * np.sqrt(Ls6 * Cs6)
            Z6 = np.sqrt(Ls6 / Cs6)
            F6 = np.array([[np.cosh(gamma6 * Lengs6), 1.0 * Z6 * np.sinh(gamma6 * Lengs6)], [1.0 / Z6 * np.sinh(gamma6 * Lengs6), np.cosh(gamma6 * Lengs6)]])
            A6, B6, C6, D6 = F6[0, 0], F6[0, 1], F6[1, 0], F6[1, 1]
            Y6 = np.array([[D6 / B6, -(A6 * D6 - B6 * C6) / B6], [-1.0 / B6, A6 / B6]])
            # 並列線路全体のY行列 ==> Sパラ, Fパラ
            Ymtx = Y1 + Y2 + Y3
            U = np.eye(2)
            Spara = np.dot((U - 50.0 * Ymtx), np.linalg.inv(U + 50.0 * Ymtx))
            # 必要な成分(S21)の抽出 (Pythonでは添え字の範囲が0からなので要注意 !!!)
            freqS21mag[n, 1] = 20.0 * np.log10(np.abs(Spara[1, 0]))
            freqS21phase[n, 1] = np.angle(Spara[1, 0])
            HR = Spara[1, 0] / (1.0 + Spara[0, 0])
            freqHRmag[n, 1] = 20.0 * np.log10(np.abs(HR))
            freqHRphase[n, 1] = np.angle(HR)


        ###############################################################
        # データ処理（位相接続して，群遅延を算出）
        ###############################################################
        pc = 0
        for xyz in range(1, len(freq)):
            if freqS21phase[xyz, 1] - freqS21phase[xyz - 1, 1] > np.pi / 2.0:
                pc = pc - 2
            if freqS21phase[xyz, 1] - freqS21phase[xyz - 1, 1] < -np.pi / 2.0:
                pc = pc + 2
            freqS21phase2[xyz, 1] = freqS21phase[xyz, 1] + np.pi * pc
        pc = 0
        for xyz in range(1, len(freq)):
            if freqHRphase[xyz, 1] - freqHRphase[xyz - 1, 1] > np.pi / 2.0:
                pc = pc - 2
            if freqHRphase[xyz, 1] - freqHRphase[xyz - 1, 1] < -np.pi / 2.0:
                pc = pc + 2
            freqHRphase2[xyz, 1] = freqHRphase[xyz, 1] + np.pi * pc

        freqS21groupdelay = -1.0 / 360.0 * np.diff(freqS21phase2[:, 1]) / np.diff(freq) * 180.0 / np.pi * 1.0e9
        freqHRgroupdelay = -1.0 / 360.0 * np.diff(freqHRphase2[:, 1]) / np.diff(freq) * 180.0 / np.pi * 1.0e9
        #NGDとＳパラと中心周波数と帯域幅


        #Smin_id2は群遅延の谷のインデックスつまり中心周波数とNGDと損失がわかる
        Smin_id=signal.argrelmin(freqS21mag)
        Smin_id2 =[]
        for l in range(len(Smin_id[0])) :
            #print(freqS21groupdelay[Smin_id[0][l]])
            if freqS21groupdelay[Smin_id[0][l]] < 0 :
                Smin_id2.append(Smin_id[0][l])
        #print(Smin_id2)

        #帯域幅を計測（freqwid)
        list = []

        freqwid = []
        for l in range(1,len(freqS21groupdelay)) :
            if freqS21groupdelay[l] <0:
                list.append(l)



        llist = [0]
        b = list[0]
        print(len(list))
        for i in range(1,len(list)) :

            if list[i]-list[i-1] >1 :

                llist.append(i-1)
                a = list[i-1] - b
                b = list[i]
                freqwid.append(a)
            if i == len(list)-1 :

                freqwid.append(list[i]-b)

                #del list[:i-1]
                #list=[i-2,i-1,i]
        freqwid2 = []
        freqwid3 = []
        freqwid4 = []
        zero = [0]
        if len(freqwid) ==3:
            tmp_se = pd.Series( freqwid, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
        elif len(freqwid) == 2:
            freqwid2 =  freqwid + zero
            tmp_se = pd.Series( freqwid2, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
        elif len(freqwid) == 1:
            freqwid3 =  freqwid + zero + zero
            tmp_se = pd.Series( freqwid3, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
        else :
            AED = ["err","err","err"]
            tmp_se = pd.Series( AED, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
            print(freqwid)

        """"
        #print(freqwid)
        S21min=[]
        SGmin=[]
        cenfre=[]
        for n in range(len(Smin_id2)):
            a= Smin_id2[n]
            S21min.append(freqS21mag[a,1])
            SGmin.append(freqS21groupdelay[a])
            cenfre.append(freqS21mag[a,0])



        NGD = S21min + SGmin + cenfre
        NGD2 = []
        NGD3 = []
        NGD4 = []
        b=[0]
        #Bの戸数はfor分で3c2-len(S21min)個でやる
        #print(freqwid)

        if len(NGD) ==15:
            tmp_se = pd.Series( NGD, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
        elif len(NGD) ==12:
            NGD2 =S21min +b+ SGmin +b+ cenfre+b
            tmp_se = pd.Series( NGD2, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
        elif len(NGD) ==9:
            NGD3 =S21min +b+b+ SGmin +b+b+ cenfre+b+b
            tmp_se = pd.Series( NGD3, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
        else :
            AED = ["err","err","err","err","err","err","err","err","err","err","err","err","err","err","err"]
            tmp_se = pd.Series( AED, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
            print(NGD)

        if len(NGD) ==9:
            tmp_se = pd.Series( NGD, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )

        elif len(NGD) ==6:
            NGD2 =S21min +b+ SGmin +b+ cenfre+b
            tmp_se = pd.Series( NGD2, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
            NGD2=[]
        elif len(NGD) ==3:

            NGD3 =S21min +b+b+ SGmin +b+b+ cenfre+b+b
            tmp_se = pd.Series( NGD3, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
            NGD3=[]
        else :
            AED = ["err",0,0,0,0,0,0,0,0]
            tmp_se = pd.Series( AED, index=list_df.columns )
            list_df = list_df.append( tmp_se, ignore_index=True )
            print(NGD)
            """
    df_concat = pd.concat([df_csv, list_df])
    df_concat.to_csv("kekka.csv")

























"""
    ###############################################################
    # Figure
    ###############################################################
    fs1 = 24  # 軸ラベルのサイズ
    fs2 = 25  # 数字サイズ
    fs3 = 18  # 凡例サイズ

    plt.figure(1, figsize=(12, 12 / 1.42))
    plt.subplot(3, 1, 1, facecolor='#FFFFFF')
    plt.gca().spines['top'].set_color("black")
    plt.gca().spines['bottom'].set_color("black")
    plt.gca().spines['left'].set_color("black")
    plt.gca().spines['right'].set_color("black")
    plt.plot(freqS21mag[:, 0] / 1e9, freqS21mag[:, 1], lw=2, alpha=0.5, linestyle="-", c='b')
    plt.xlim(1.6, 1.75)
    plt.ylim(-15, 0)
    #plt.xticks([1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0], [])
    plt.ylabel('$|S_{21}|$ [dB]', fontsize=fs1)
    plt.tick_params(labelsize=fs2)
    plt.tight_layout()
    plt.subplot(3, 1, 2, facecolor='#FFFFFF')
    plt.gca().spines['top'].set_color("black")
    plt.gca().spines['bottom'].set_color("black")
    plt.gca().spines['left'].set_color("black")
    plt.gca().spines['right'].set_color("black")
    plt.plot(freqS21phase[:, 0] / 1e9, freqS21phase[:, 1], lw=2, alpha=0.5, linestyle="-", c='b', label="wrapped")
    plt.plot(freqS21phase2[:, 0] / 1e9, freqS21phase2[:, 1], lw=3, alpha=0.5, linestyle=":", c='r', label="unwrapped")
    plt.xlim(1.6, 1.75)
    plt.ylim(-2.0 * np.pi, 2.0 * np.pi)
    #plt.xticks([1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0], [])
    plt.yticks([-2.0 * np.pi, -np.pi, 0, np.pi, 2.0 * np.pi], ['$-2\\pi$', '$-\\pi$', '0', '$\\pi$', '$2\\pi$'])
    plt.ylabel('Phase of $S_{21}$ [rad]', fontsize=fs1)
    leg = plt.legend(loc='upper right', fontsize=fs3, ncol=2, frameon=True, fancybox=True, shadow=True)
    leg.get_frame().set_alpha(0.8)
    plt.tick_params(labelsize=fs2)
    plt.tight_layout()

    plt.subplot(3, 1, 3, facecolor='#FFFFFF')
    plt.gca().spines['top'].set_color("black")
    plt.gca().spines['bottom'].set_color("black")
    plt.gca().spines['left'].set_color("black")
    plt.gca().spines['right'].set_color("black")
    plt.plot(freq2 / 1e9, freqS21groupdelay, lw=2, alpha=0.5, linestyle="-")
    plt.xlim(1.6, 1.75)
    plt.ylim(-10, 5)
    #plt.xticks([1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0],[1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0])
    plt.xlabel('Frequency [GHz]', fontsize=fs1)
    plt.ylabel('Group delay $\\tau_{g}$ [ns]', fontsize=fs1)
    plt.tick_params(labelsize=fs2)
    plt.tight_layout()


    plt.savefig('Results-NGD_S21t1=0.26t2=0.31t3=0.265.pdf', format='pdf', dpi=150, transparent=True, bbox_inches='tight', pad_inches=0.0)
    #plt.savefig('Results-NGD_S21t1=0.26t2=0.31t3=0.36.pdf', format='pdf', dpi=150, transparent=True, bbox_inches='tight', pad_inches=0.0)
    #plt.savefig("Results-NGD_S21t1=0.26t2=0.31-1.png")



    plt.show()
"""
