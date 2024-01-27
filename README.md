# MPFAD
MPFAD(Multi Precision Fast Automatic Diffrentiation)はMPFRおよびMPFR C++を利用した任意精度自動微分ライブラリです. 
高階微分を求めることも可能ですが, 現状はTop Down算法(Reverse-mode AD)のみ実装しています. 

# How to use
C++17で動作確認しています. 
詳細はexample.cppを参照してください. 

# Installation
MPFRのインストールはMPFRのマニュアル (https://www.mpfr.org/mpfr-4.0.2/mpfr.html#Installing-MPFR) を参照してください.
GCCを使う場合はMPFADはMakefileで導入できます.
```
$ make clean
$ make .c.o
$ make
```
# License
このライブラリは GPL-3.0ライセンスの元にライセンスされています. 詳細はLICENSE.mdを確認してください.
