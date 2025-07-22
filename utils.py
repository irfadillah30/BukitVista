import re

def extract_facilities(facility_str):
    tamu = re.search(r'(\d+)\s+tamu', facility_str)
    kamar_tidur = re.search(r'(\d+)\s+kamar tidur', facility_str)
    tempat_tidur = re.search(r'(\d+)\s+tempat tidur', facility_str)
    kamar_mandi = re.search(r'(\d+)\s+kamar mandi', facility_str)

    return [
        int(tamu.group(1)) if tamu else 0,
        int(kamar_tidur.group(1)) if kamar_tidur else 0,
        int(tempat_tidur.group(1)) if tempat_tidur else 0,
        int(kamar_mandi.group(1)) if kamar_mandi else 0
    ]
