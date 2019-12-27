from collections import OrderedDict


"""
OAI iMorphics reference classes (DICOM attribute names). Cartilage tissues
"""
locations_mh53 = OrderedDict([
    ('Background', 0),
    ('FemoralCartilage', 1),
    ('LateralTibialCartilage', 2),
    ('MedialTibialCartilage', 3),
    ('PatellarCartilage', 4),
    ('LateralMeniscus', 5),
    ('MedialMeniscus', 6),
])


"""
Segmentation predictions. Major cartilage tissues. Joined L and M
"""
locations_f43h = OrderedDict([
    ('_background', 0),
    ('femoral', 1),
    ('tibial', 2),
])


"""
Segmentation predictions. Cartilage tissues. Joined L and M
"""
locations_zp3n = OrderedDict([
    ('_background', 0),
    ('femoral', 1),
    ('tibial', 2),
    ('patellar', 3),
    ('menisci', 4),
])


atlas_to_locations = {
    'imo': locations_mh53,
    'segm': locations_zp3n,
    'okoa': locations_f43h,
}
