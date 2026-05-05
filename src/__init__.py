import os
import sys

# Ensure the RDF package root is on sys.path so imports like `src.core...` work
_rdf_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _rdf_root not in sys.path:
    sys.path.append(_rdf_root)
