#!/usr/bin/env python3
# -*- coding: utf-8 -*-"""
###############################################################
"""
@author: kayano
"""
import matplotlib.pyplot as plt
import numpy as np

"""
グラフやフォント等の設定bbbbbbbbhb
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
"""これは一つの分散型遅延線路のプログラムです"""
###############################################################
"""メイン関数"""
if __name__ == '__main__':


    

    sum_Lengs = 90

    Alt_Lengs1 = 42.18111736219896
    Alt_Lengs2 = 57.721529021956464



    # 解析周波数範囲
    delta_f = 0.0001e9
    fmin = 2.0e9

    fmax = 4.0e9


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
    Lengs1 = Alt_Lengs1*1.0e-3
    #Lengs1 = vp1 * 0.26e-9
    Rs1 = alpha_tgt * 2 * Z1 / (8.686 * Lengs1)

    """ 線路 2 """
    Cs2 = 7.79498e-11
    Ls2 = 4.00713e-7
    Z2 = np.sqrt(Ls2 / Cs2)
    vp2 = 1.0 / np.sqrt(Ls2 * Cs2)
    Lengs2 = Alt_Lengs2*1.0e-3
    #Lengs2 = vp2 * 0.28e-9
    Rs2 = alpha_tgt * 2 * Z2 / (8.686 * Lengs2)

    #print(Rs2)
    print("t1="+str(Lengs1/vp1),"t2="+str(Lengs2/vp2))
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

        # 並列線路全体のY行列 ==> Sパラ, Fパラ
        Ymtx = Y1 + Y2
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
    SGmin=np.min(freqS21groupdelay[1:])
    print("Sパラのこと")
    print("NGD"+str(SGmin))
    Smin = np.min(freqS21mag[1:])
    print("Sパラの損失"+str(Smin))
    print("中心周波数"+str(np.argmin(freqS21groupdelay[1:])*0.0001e9+fmin))
    x = np.where(freqS21groupdelay < 0)#1
    #print("帯域幅"+str((np.max(x)-np.min(x))*0.0001e9))#2
    #print(str(np.max(x))+"-"+str(np.min(x)))
    #print("性能"+str((np.max(x)-np.min(x))*0.0001e9*SGmin))
    
    ###############################################################
    # Figure
    ###############################################################

    fs1 = 24  # 軸ラベルのサイズ
    fs2 = 25  # 数字サイズ
    fs3 = 18  # 凡例サイズ
    xlimit1 = fmin
    xlimit2 = fmax

    #xlimit1 = fmin*1.0e9
    #xlimit2 = fmax*1.0e9
    plt.figure(1, figsize=(12, 12 / 1.42))
    plt.subplot(3, 1, 1, facecolor='#FFFFFF')
    plt.gca().spines['top'].set_color("black")
    plt.gca().spines['bottom'].set_color("black")
    plt.gca().spines['left'].set_color("black")
    plt.gca().spines['right'].set_color("black")
    plt.plot(freqS21mag[:, 0] / 1e9, freqS21mag[:, 1], lw=2, alpha=0.5, linestyle="-", c='b')
    #plt.xlim(xlimit1, xlimit2)
    #plt.ylim(-10, 0)
    plt.xticks([1.70, 1.72, 1.74, 1.76, 1.78, 1.80], [])
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
    #plt.xlim(xlimit1, xlimit2)
    plt.ylim(-2.0 * np.pi, 2.0 * np.pi)
    plt.xticks([1.70, 1.72, 1.74, 1.76, 1.78, 1.80], [])
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
    #plt.xlim(xlimit1, xlimit2)
    #plt.ylim(-100, 5)
    #plt.xticks([1.70, 1.72, 1.74, 1.76, 1.78, 1.80], [1.70, 1.72, 1.74, 1.76, 1.78, 1.80])
    plt.xlabel('Frequency [GHz]', fontsize=fs1)
    plt.ylabel('Group delay $\\tau_{g}$ [ns]', fontsize=fs1)
    plt.tick_params(labelsize=fs2)
    plt.tight_layout()

    #plt.savefig('Results-NGD.pdf', format='pdf', dpi=150, transparent=True, bbox_inches='tight', pad_inches=0.0)
    #plt.savefig("Results-NGD_S21t1=0.26t2=0.31-1.png")

  

    plt.show()
