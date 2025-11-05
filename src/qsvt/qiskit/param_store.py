# param_store.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json
from collections import defaultdict
import pennylane as qml

EPSIRON = 0.01
KAPPA_LIST = [5, 50, 100, 250, 500, 1000, 1500]

DEFAULT_OUTPUT = Path(__file__).parent / "angles_qsvt.json"


class ParamStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: defaultdict[float, Dict[float, Any]] = defaultdict(dict)

    # --- dict風API ---
    def __getitem__(self, epsilon: float) -> Dict[float, Any]:
        return self._data[epsilon]

    def __setitem__(self, epsilon: float, inner: Dict[float, Any]) -> None:
        # 内側辞書はそのまま（丸めなし）
        self._data[epsilon] = dict(inner)

    # 便利メソッド
    def set(self, epsilon: float, kappa: float, value: Any) -> None:
        self._data[epsilon][kappa] = value

    def get(self, epsilon: float, kappa: float, default: Any = None) -> Any:
        return self._data.get(epsilon, {}).get(kappa, default)

    # --- 永続化 ---
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ser = {str(eps): {str(kap): v for kap, v in inner.items()}
                for eps, inner in self._data.items()}
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(ser, f, ensure_ascii=False, indent=2)



def save_data(
    path: str | Path = DEFAULT_OUTPUT,
    epsilon: float = EPSIRON,
    kappas: List[int] = KAPPA_LIST,
) -> None:

    # PennyLane の inverse データセットを読み込み
    [dataset] = qml.data.load("other", name="inverse")

    epsilon_key = f"{epsilon:.2f}"

    store = ParamStore(path)

    for kappa in kappas:
        print(f"Saving data for epsilon={epsilon}, kappa={kappa}")
        kappa_key = str(kappa)
        try:
            angles = dataset.angles["qsvt"][epsilon_key][kappa_key]
        except KeyError as exc:
            raise KeyError(
                f"Angles not found for epsilon={epsilon_key}, kappa={kappa_key}."
            ) from exc

        # JSONに保存できるようリストへ変換
        store.set(epsilon, float(kappa), [float(a) for a in angles])

    store.save()


def load_angles(
    epsilon: float,
    kappa: float,
) -> List[float]:
    with open(DEFAULT_OUTPUT, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw[str(epsilon)][str(kappa)]


if __name__ == "__main__":
    save_data()
