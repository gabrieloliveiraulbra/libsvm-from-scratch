# antigravity.svm

Uma implementação puramente em **NumPy** de Máquinas de Vetores de Suporte (Support Vector Machines) baseada na biblioteca LIBSVM.

> Chang, C.-C., & Lin, C.-J. (2011). **LIBSVM: A library for support vector machines**.  
> *ACM Transactions on Intelligent Systems and Technology*, 2(3), 27:1–27:27.

---

## Funcionalidades

| Módulo | Conteúdo |
|--------|----------|
| `antigravity.svm.kernels` | Linear, Polinomial, RBF, Sigmoide |
| `antigravity.svm.solver` | SMO + WSS-3 + LRU cache + Shrinking + KKT |
| `antigravity.svm.models` | C-SVC, ν-SVC, One-Class SVM, ε-SVR, ν-SVR |
| `antigravity.svm.multiclass` | Um-Contra-Um (One-Against-One, k(k-1)/2 classificadores binários) |
| `antigravity.svm.probability` | Escalonamento de Platt + Intervalos Laplace SVR |
| `antigravity.svm.tuning` | Busca em Grade (Grid Search) + Validação Cruzada k-fold |

**Dependências**: Apenas `numpy >= 1.21` (sem scipy, sem sklearn).

---

## Visualizador Web Interativo ✨ NOVO

O projeto agora inclui um **Visualizador Web Full-Stack** para interagir e visualizar em tempo real as fronteiras de decisão da SVM. O frontend foi construído com React e Canvas HTML5 puro, e o backend usa FastAPI para processar os cálculos do SVM.

### Como iniciar o Visualizador

**Passo 1: Iniciar o Backend (FastAPI)**
Abra um terminal, vá até a pasta do backend e inicie o servidor:
```bash
cd web-app/backend
pip install fastapi uvicorn pydantic
python -m uvicorn main:app --port 8000
```

**Passo 2: Iniciar o Frontend (React/Vite)**
Abra outro terminal, vá até a pasta do frontend, instale as dependências e inicie o site:
```bash
cd web-app/frontend
npm install
npm run dev
```

Acesse o link gerado (geralmente `http://localhost:5173`) no seu navegador e divirta-se criando pontos e vendo a Inteligência Artificial separar as classes!

---

## Instalação (Uso como Biblioteca)

```bash
pip install -e .
```

---

## Início Rápido

### Classificação Binária (C-SVC)

```python
import numpy as np
from antigravity.svm import CSVC, KERNEL_RBF

X_train = np.random.randn(100, 2)
y_train = np.sign(X_train[:, 0])

clf = CSVC(C=1.0, kernel=KERNEL_RBF, gamma=0.5)
clf.fit(X_train, y_train)
print(clf.predict(X_train[:5]))
```

### Multiclasse (Um-Contra-Um)

```python
from antigravity.svm import MulticlassSVC

mc = MulticlassSVC(C=5.0, kernel=KERNEL_RBF, gamma=0.3)
mc.fit(X_train, y_multiclass)
print(mc.predict(X_test))
```

### Busca em Grade (Grid Search)

```python
from antigravity.svm import CSVC, GridSearchCV, KERNEL_RBF

gs = GridSearchCV(
    CSVC,
    {"C": [0.1, 1.0, 10.0], "gamma": [0.01, 0.1, 1.0]},
    cv=5,
    scoring="accuracy",
    fixed_params={"kernel": KERNEL_RBF},
)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
print(gs.summary())
```

### Calibração de Probabilidades (Escalonamento de Platt)

```python
from antigravity.svm.probability import PlattScaling

platt = PlattScaling(n_folds=5)
platt.fit(clf.decision_function(X_train), y_train.astype(float))
proba = platt.predict_proba(clf.decision_function(X_test))
```

### Regressão (ε-SVR) com Intervalos de Previsão

```python
from antigravity.svm import EpsilonSVR
from antigravity.svm.probability import LaplaceSVR

reg = EpsilonSVR(C=10.0, epsilon=0.1, kernel=KERNEL_RBF, gamma=0.5)
reg.fit(X_train, y_train)

lap = LaplaceSVR(n_folds=5)
lap.fit_with_cv(lambda: EpsilonSVR(C=10.0, epsilon=0.1), X_train, y_train)
lower, upper = lap.predict_interval(reg.predict(X_test), confidence=0.95)
```

---

## Executando Testes

```bash
pytest tests/ -v
```

## Executando a Demo Simples no Terminal

```bash
python examples/demo.py
```

---

## Formulações Matemáticas

### Dual C-SVC (Eq. 1)

```
min   ½ αᵀ Q α − eᵀ α
s.t.  yᵀ α = 0,   0 ≤ αᵢ ≤ C
      Q_ij = y_i y_j K(x_i, x_j)
```

### Dual ν-SVC (Eq. 3)

```
min   ½ ᾱᵀ Q ᾱ
s.t.  yᵀ ᾱ = 0,   eᵀ ᾱ = ν,   0 ≤ ᾱᵢ ≤ 1/l
```

### Dual ε-SVR (Eq. 5)

```
min   ½ (α−α*)ᵀ Q (α−α*) + ε eᵀ(α+α*) − yᵀ(α−α*)
s.t.  eᵀ(α−α*) = 0,   0 ≤ αᵢ, αᵢ* ≤ C
```

### SMO / WSS-3

O algoritmo de resolução usa **seleção de conjunto de trabalho de segunda ordem** (Fan, Chen & Lin, 2005):

1. Selecione *i* que maximiza −yᵢ ∇fᵢ em I_up
2. Selecione *j* que minimiza o decréscimo quadrático:
   `− (∇f_i − ∇f_j)² / (Q_ii + Q_jj − 2 Q_ij)`

A convergência é declarada quando a diferença das condições KKT (KKT gap) atinge:

```
max_{I_up}(−y·∇f) − min_{I_low}(−y·∇f) ≤ ε
```

---

## Licença

MIT
