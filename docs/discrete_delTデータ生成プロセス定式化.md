# discreteなDelTを用いたCTMCデータ生成プロセスの定式化

## 1. 目的
本ドキュメントは、`utils/data_generator_discrete.py` で実装されている **discreteなDelT（観測間隔）** を用いたデータ生成処理を、数理的に定式化しつつ実装手順に沿って整理したものである。

## 2. 前提
状態数を $$N$$ とし、状態集合を
$$
\mathcal{S}=\{1,2,\dots,N\}
$$
とする。

本実装では、劣化方向のみの遷移（上三角の隣接遷移）を仮定し、推移率行列は
$$
\boldsymbol{Q}=(q_{ij})\in\mathbb{R}^{N\times N}
$$
で表す。

## 3. 推移率行列の生成
各状態 $$i\in\{1,\dots,N-1\}$$ に対して寿命上限パラメータ $$L$$ を用い、
$$
\nu_i\sim \mathrm{Uniform}(1,L)
$$
からサンプルし、
$$
\lambda_i=\frac{1}{\nu_i}
$$
とする。

このとき、行列要素は
$$
q_{ii}=-\lambda_i,\quad q_{i,i+1}=\lambda_i,\quad q_{ij}=0\ (j\notin\{i,i+1\})
$$
とする。最終状態 $$N$$ は吸収状態として
$$
q_{Nj}=0\ (\forall j)
$$
となる。

## 4. discreteなDelTの生成（Dirichlet混合）
### 4.1 離散候補点の構築
まず候補数 $$K$$ を決める。実装では
$$
K\sim \mathrm{DiscreteUniform}(K_{\min},K_{\max})
$$
（既定値: $$K_{\min}=2,\ K_{\max}=10$$）である。

次に候補時刻
$$
\tau_1,\tau_2,\dots,\tau_K\sim \mathrm{Uniform}(1,100)
$$
を生成する。

### 4.2 候補点への確率重み付け
Dirichlet分布で重み
$$
\boldsymbol{w}=(w_1,\dots,w_K)\sim \mathrm{Dirichlet}(\mathbf{1}_K)
$$
を生成する。

### 4.3 DelTのサンプリング
各サンプルでインデックス
$$
Z\sim \mathrm{Categorical}(w_1,\dots,w_K)
$$
を引き、
$$
\Delta t=\tau_Z
$$
とする。実装ではさらに
$$
\Delta t\leftarrow \mathrm{round}(\Delta t,1)
$$
で小数第1位丸めを行う。

以上により、DelTは連続分布ではなく、有限集合
$$
\{\tau_1,\dots,\tau_K\}
$$
上の離散分布として扱われる。

## 5. 初期状態と遷移サンプルの生成
### 5.1 初期状態分布
初期状態は最終状態を除く集合
$$
\{1,\dots,N-1\}
$$
で生成する。まず
$$
\boldsymbol{\pi}=(\pi_1,\dots,\pi_{N-1})\sim \mathrm{Dirichlet}(\mathbf{1}_{N-1})
$$
を作り、
$$
S_0\sim \mathrm{Categorical}(\pi_1,\dots,\pi_{N-1})
$$
で初期状態をサンプルする。

### 5.2 観測間隔後の状態
与えられた $$\Delta t$$ に対し、CTMCの遷移確率行列
$$
\boldsymbol{P}(\Delta t)=\exp(\boldsymbol{Q}\Delta t)
$$
を計算する。

次状態 $$S_1$$ は
$$
S_1\sim \mathrm{Categorical}(P_{S_0,1}(\Delta t),\dots,P_{S_0,N}(\Delta t))
$$
で生成する。

1サンプルは
$$
\boldsymbol{x}=(S_0,S_1,\Delta t)
$$
で表す。

## 6. 1データセット生成アルゴリズム
データセットサイズを $$M$$ とすると、以下の手順で生成する。

1. $\boldsymbol{Q}$ を生成する。  
2. Dirichlet混合により discreteなDelT生成器を初期化する。  
3. $$m=1,\dots,M$$ について、$$S_0^{(m)},\Delta t^{(m)},S_1^{(m)}$$ を順にサンプルし、
   $$
   \boldsymbol{x}^{(m)}=(S_0^{(m)},S_1^{(m)},\Delta t^{(m)})
   $$
   を蓄積する。  
4. 先頭に $$\boldsymbol{Q}$$ 行列を付与し、CSVに保存する。  
5. 追加で尤度法による推定結果（`_insert_likelihood_results`）を挿入し、教師信号として利用可能な行を拡張する。

## 7. 乱数シードと並列生成
データセット番号を $$d$$、ベースシードを $$s_0$$ とすると、子シードは
$$
s_d=\mathrm{SeedSequence}(s_0,d)
$$
で決定する。

これにより、並列実行時でもデータセットごとの再現性を保ちながら
$$
\{\mathcal{D}_1,\dots,\mathcal{D}_C\}
$$
（$$C$$ 個）を独立生成できる。

## 8. 実装対応表
- discreteなDelT生成: `DirichletDeltaT`  
- 行列生成: `DiagonalTransitionRateMatrixGenerator`  
- サンプル生成: `DataGenerator.generate_matrix`  
- 遷移確率計算: `CalcProbmatrix`  
- 並列化: `_run_parallel_from_args`

## 9. 実運用上の補足
- DelT候補点はデータセットごとに固定されるため、同一データセット内で「同じ観測間隔」が繰り返し出現しやすい。  
- Dirichlet重みは候補点の出現頻度を非一様化できるため、単純な離散一様サンプリングより柔軟である。  
- 候補点範囲 $$[1,100]$$ と丸め処理は、学習データ分布のバイアスに直結するため、推論対象データの観測間隔分布との整合性を必ず確認する。
