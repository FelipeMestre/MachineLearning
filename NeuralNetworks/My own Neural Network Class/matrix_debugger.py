import numpy as np

try:
    from rich.console import Console
except ImportError:
    Console = None


class _MatrixDebugger:
    """
    Pretty-printer opcional para matrices durante backprop.
    - Si `rich` está instalado: salida con estilos.
    - Si no: fallback a `print`.
    """

    def __init__(self, max_rows=8, max_cols=8, precision=4):
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.precision = precision
        self._console = Console() if Console is not None else None

    def _emit(self, text: str) -> None:
        if self._console is not None:
            self._console.print(text)
        else:
            print(text)

    def title(self, text: str) -> None:
        if self._console is not None:
            self._emit(f"\n[bold cyan]{text}[/bold cyan]")
        else:
            self._emit(f"\n{text}")

    def _slice_view(self, a: np.ndarray) -> tuple[np.ndarray, str]:
        if a is None:
            return a, ""
        if a.ndim == 0:
            return a, ""
        if a.ndim == 1:
            view = a[: self.max_cols]
            suffix = "" if a.shape[0] <= self.max_cols else f"  (mostrando 0:{self.max_cols})"
            return view, suffix
        if a.ndim == 2:
            r = min(self.max_rows, a.shape[0])
            c = min(self.max_cols, a.shape[1])
            view = a[:r, :c]
            suffix = ""
            if a.shape[0] > r or a.shape[1] > c:
                suffix = f"  (mostrando 0:{r}, 0:{c})"
            return view, suffix
        # Tensores >2D: mostramos una vista pequeña para evitar inundar la salida.
        # Casos comunes:
        # - (batch_size, height, width, channels)  -> fijamos batch=0 y recortamos height/width/canales.
        # - (height, width, channels)             -> recortamos height/width/canales.
        if a.ndim == 3:
            r = min(self.max_rows, a.shape[0])
            c = min(self.max_cols, a.shape[1])
            k = min(3, a.shape[2])
            view = a[:r, :c, :k]
            suffix = ""
            if a.shape[0] > r or a.shape[1] > c or a.shape[2] > k:
                suffix = f"  (mostrando 0:{r}, 0:{c}, 0:{k})"
            return view, suffix

        if a.ndim == 4:
            r = min(self.max_rows, a.shape[1])
            c = min(self.max_cols, a.shape[2])
            k = min(3, a.shape[3])
            view = a[0, :r, :c, :k]
            suffix = ""
            if a.shape[0] > 1 or a.shape[1] > r or a.shape[2] > c or a.shape[3] > k:
                suffix = f"  (mostrando batch=0, 0:{r}, 0:{c}, 0:{k})"
            return view, suffix

        # Fallback genérico: fijamos a 0 todas las dimensiones excepto las dos últimas.
        r = min(self.max_rows, a.shape[-2])
        c = min(self.max_cols, a.shape[-1])
        leading_index = (0,) * (a.ndim - 2)
        view = a[leading_index + (slice(0, r), slice(0, c))]
        suffix = ""
        if a.shape[-2] > r or a.shape[-1] > c or any(dim > 1 for dim in a.shape[:-2]):
            suffix = f"  (mostrando indices iniciales, 0:{r}, 0:{c})"
        return view, suffix

    def array(self, name: str, a: np.ndarray) -> None:
        if a is None:
            self._emit(f"- {name}: None")
            return

        view, suffix = self._slice_view(a)
        shape = getattr(a, "shape", None)
        stats = ""
        try:
            if np.issubdtype(a.dtype, np.number):
                stats = (
                    f"  min={np.min(a):.{self.precision}g}"
                    f" max={np.max(a):.{self.precision}g}"
                    f" mean={np.mean(a):.{self.precision}g}"
                )
        except Exception:
            stats = ""

        try:
            if isinstance(view, np.ndarray) and view.ndim <= 3:
                body = np.array2string(
                    view,
                    precision=self.precision,
                    suppress_small=True,
                    max_line_width=120,
                )
            else:
                body = repr(view)
        except Exception:
            body = repr(view)

        if self._console is not None:
            self._emit(f"[bold]{name}[/bold] shape={shape}{stats}{suffix}\n{body}\n")
        else:
            self._emit(f"{name} shape={shape}{stats}{suffix}\n{body}\n")


# Alias público (sin guion bajo) para usarlo desde otros módulos sin “clase privada”.
MatrixDebugger = _MatrixDebugger
