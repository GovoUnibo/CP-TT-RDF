# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
import torch



class RDF_TDBase(RDFEngine):
    """
    Base class per modelli RDF basati su decomposizioni tensoriali (CP, TT, ...).
    Contiene solo utilità condivise da CP e TT; nessun dettaglio di fattorizzazione.
    """
    def __init__(self, dtype=torch.float32, device='cpu'):
        super().__init__(dtype=dtype, device=device)
        self._td_ready: bool = False   # flag: fattori caricati
        self.model_extension: str = "" # già usato nel resto del framework, se serve

    # ------------------------------------------------------------
    # Helpers condivisi
    # ------------------------------------------------------------
    def _init_domain_from_first_link(self, link_names: List[str]):
        """Inizializza dominio/scala/offset (e, se serve, ellissoide) dal primo link."""
        m0: Model = getattr(self, link_names[0] + self.model_extension)
        super().set_points_domain(m0.domain_min, m0.domain_max)
        # NB: scale/centroid su batch verranno impostati via set_ordered_batch_params
        if getattr(self, "method", "sphere") == "ellipsoid" and (m0.center_ellipsoid is not None):
            super().set_ellipsoid_parameters(
                m0.center_ellipsoid, m0.axes_ellipsoid, m0.eigen_vector_ellipsoid
            )

    def _normalize_ranks_input(self, link_names: List[str], ranks_per_link):
        """
        Converte ranks_per_link in lista di int allineata a link_names.
        - int -> [int]*L
        - list/tuple -> stessa lunghezza
        - dict -> mappa per nome
        """
        L = len(link_names)
        if isinstance(ranks_per_link, int):
            return [int(ranks_per_link)] * L
        if isinstance(ranks_per_link, (list, tuple)):
            assert len(ranks_per_link) == L, f"Attesi {L} rank, ricevuti {len(ranks_per_link)}"
            return [int(r) for r in ranks_per_link]
        if isinstance(ranks_per_link, dict):
            try:
                return [int(ranks_per_link[name]) for name in link_names]
            except KeyError as e:
                missing = str(e).strip("'")
                raise KeyError(f"Rank mancante per link '{missing}' in ranks_per_link") from None
        raise TypeError("ranks_per_link deve essere int | list[int] | dict{name:int}")
