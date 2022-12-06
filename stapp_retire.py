### 事前準備 ###
#
# AiPro2022フォルダに移動して…
#
# > py  -3.9  -m  venv  .scikit     ← 新しい仮想環境「.scikit」を生成する
#
# > .scikit\Scripts\activate.bat    ← 新しい仮想環境「.scikit」を起動する
#
# (.scikit) > python -m pip install -r requirements_scikit.txt   ← ライブラリの一括インストール
#
# ※一括インストールは時間がかかりますので(10分～？）休み時間の前などに実行するとよいでしょう
#
#
# requirements_scikit.txt の内容
#   streamlit               ← 毎度、おなじみ
#   typing_extensions       ← これがないとエラーが出ることがあるので一応
#   numpy                   ← 毎度、おなじみ
#   pandas                  ← 毎度、おなじみ
#   scikit-learn            ← 超有名、機械学習ライブラリ 
#   imblearn                ← データの水増し用
#   matplotlib              ← 超有名、グラフ描画ライブラリ
#   japanize-matplotlib     ← （の日本語対応版）
#   seaborn                 ← こちらもメジャーなグラフ描画ライブラリ
#
#
# (.tfflow) > streamlit run stapp_retire.py    ← Webアプリ（stapp_retire.py）を起動


""" Streamlitによる退職予測AIシステムの開発
"""

from itertools import chain
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import japanize_matplotlib
import seaborn as sns 

# 決定木
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

# 精度評価用
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# データを分割するライブラリを読み込む
from sklearn.model_selection import train_test_split

# データを水増しするライブラリを読み込む
from imblearn.over_sampling import SMOTE

# ロゴの表示用
from PIL import Image

# ディープコピー
import copy

sns.set()
japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def st_display_table(df: pd.DataFrame):
    """
    Streamlitでデータフレームを表示する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        対象のデータフレーム

    Returns
    -------
    なし
    """

    # データフレームを表示
    st.subheader('データの確認')
    st.table(df)

    # 参考：Streamlitでdataframeを表示させる | ITブログ
    # https://kajiblo.com/streamlit-dataframe/


def st_display_graph(df: pd.DataFrame, x_col : str):
    """
    Streamlitでグラフ（ヒストグラム）を表示する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        対象のデータフレーム
    x_col : str
        対象の列名（グラフのx軸）

    Returns
    -------
    なし
    """

    fig, ax = plt.subplots()    # グラフの描画領域を準備
    plt.grid(True)              # 目盛線を表示する

    # グラフ（ヒストグラム）の設定
    sns.countplot(data=df, x=x_col, ax=ax)

    st.pyplot(fig)              # Streamlitでグラフを表示する


def ml_dtree(
    X: pd.DataFrame,
    y: pd.Series,
    depth: int) -> list:
    """
    決定木で学習と予測を行う関数
    
    Parameters
    ----------
    X : pd.DataFrame
        説明変数の列群
    y : pd.Series
        目的変数の列
    depth : int
        決定木の深さ

    Returns
    -------
    list: [学習済みモデル, 予測値, 正解率]
    """

    # 決定木モデルの生成（オプション:木の深さ）
    clf = DecisionTreeClassifier(max_depth=depth)

    # 学習
    clf.fit(X, y)

    # 予測
    pred = clf.predict(X)

    # accuracyで精度評価
    score = accuracy_score(y, pred)

    return [clf, pred, score]


def st_display_dtree(clf, features):
    """
    Streamlitで決定木のツリーを可視化する関数
    
    Parameters
    ----------
    clf : 
        学習済みモデル
    features :
        説明変数の列名

    Returns
    -------
    なし
    """

    # 可視化する決定木の生成
    dot = tree.export_graphviz(clf, 
        out_file=None,          # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
        filled=True,            # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
        rounded=True,           # Trueにすると、ノードの角を丸く描画する。
        feature_names=features, # これを指定しないとチャート上で説明変数の名前が表示されない
        special_characters=True # 特殊文字を扱えるようにする
        )

    # Streamlitで決定木を表示する
    st.graphviz_chart(dot)


def main():
    """ メインモジュール
    """

    # stのタイトル表示
    st.title("退職予測AI\n（Maschine Learning)")

    # サイドメニューの設定
    activities = ["データ確認", "要約統計量", "グラフ表示", "学習と検証", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'データ確認':

        # ファイルのアップローダー
        uploaded_file = st.sidebar.file_uploader("訓練用データのアップロード", type='csv') 

        # アップロードの有無を確認
        if uploaded_file is not None:

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字列の判定
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み
                df = pd.read_csv(uploaded_file, encoding=enc) 

                # データフレームをセッションステートに退避（名称:df）
                st.session_state.df = copy.deepcopy(df)

                # スライダーの表示（表示件数）
                cnt = st.sidebar.slider('表示する件数', 1, len(df), 10)

                # テーブルの表示
                st_display_table(df.head(int(cnt)))

        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == '要約統計量':

        # セッションステートにデータフレームがあるかを確認
        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            # 要約統計量の表示

            
        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == 'グラフ表示':

        # セッションステートにデータフレームがあるかを確認
        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            # グラフの表示

            
        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == '学習と検証':

        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)

            # 説明変数と目的変数の設定
            train_X = df.drop("退職", axis=1)   # 退職列以外を説明変数にセット
            train_Y = df["退職"]                # 退職列を目的変数にセット

            # 決定木による予測
            clf, train_pred, train_scores = ml_dtree(train_X, train_Y, 2)

            # 正解率を出力


            # 決定木のツリーを出力
            

        else:
            st.subheader('訓練用データをアップロードしてください')
        

if __name__ == "__main__":
    main()

