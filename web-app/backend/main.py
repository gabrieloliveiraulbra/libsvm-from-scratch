from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import sys
import os

# Ensure antigravity can be imported from parent directory BEFORE the standard library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from antigravity.svm import CSVC, KERNEL_LINEAR, KERNEL_RBF, KERNEL_POLY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    points: list[list[float]] # [[x1, y1], [x2, y2]]
    labels: list[int]         # [1, -1, 1, ...]
    C: float = 1.0
    kernel: str = 'RBF'       # 'LINEAR', 'RBF', 'POLY'
    gamma: float = 1.0
    degree: int = 3

@app.post("/api/train")
def train_svm(request: TrainRequest):
    if not request.points or len(request.points) < 2:
        return {"error": "Need at least 2 points to train"}
    
    X = np.array(request.points)
    y = np.array(request.labels)
    
    # Must have both classes to train SVC properly
    if len(np.unique(y)) < 2:
        return {"error": "Need points of both classes (+1 and -1) to train CSVC"}
        
    kernel_map = {
        'LINEAR': KERNEL_LINEAR,
        'RBF': KERNEL_RBF,
        'POLY': KERNEL_POLY
    }
    
    kernel_func = kernel_map.get(request.kernel.upper(), KERNEL_RBF)
    
    clf = CSVC(C=request.C, kernel=kernel_func, gamma=request.gamma, degree=request.degree)
    
    try:
        clf.fit(X, y)
    except Exception as e:
        return {"error": str(e)}
        
    # Generate decision boundary grid
    # Determine bounds with a little padding
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    
    # Using a 50x50 grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
                         
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = clf.decision_function(grid_points)
    Z = Z.reshape(xx.shape)
    
    support_vectors = clf.support_.tolist() if clf.support_ is not None else []
    
    return {
        "x": xx[0, :].tolist(),
        "y": yy[:, 0].tolist(),
        "z": Z.tolist(),
        "support_vectors": support_vectors,
        "intercept": float(clf.intercept_) if clf.intercept_ is not None else 0.0,
        "bounds": [float(x_min), float(x_max), float(y_min), float(y_max)]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
