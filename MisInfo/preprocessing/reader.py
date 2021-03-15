import json
from typing import List

from preprocessing.feature_eng import Datum


def read_json_data(datapath: str) -> List[Datum]:
    with open(datapath) as f:
        data = json.load(f)
        return [Datum(**point) for point in data]
