import re

def extract_facilities(facility_str):
    jumlah_tamu = re.search(r'(\d+)\s+tamu', facility_str)
    jumlah_kamar_tidur = re.search(r'(\d+)\s+kamar tidur', facility_str)
    jumlah_tempat_tidur = re.search(r'(\d+)\s+tempat tidur', facility_str)
    jumlah_kamar_mandi = re.search(r'(\d+)\s+kamar mandi', facility_str)

    return [
        int(jumlah_tamu.group(1)) if jumlah_tamu else 0,
        int(jumlah_kamar_tidur.group(1)) if jumlah_kamar_tidur else 0,
        int(jumlah_tempat_tidur.group(1)) if jumlah_tempat_tidur else 0,
        int(jumlah_kamar_mandi.group(1)) if jumlah_kamar_mandi else 0,
    ]
