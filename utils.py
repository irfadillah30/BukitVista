import re

def extract_facilities(facility_str):
    guests = re.search(r'(\d+) tamu', facility_str)
    bedrooms = re.search(r'(\d+) kamar tidur', facility_str)
    beds = re.search(r'(\d+) tempat tidur', facility_str)

    guests = int(guests.group(1)) if guests else 0
    bedrooms = int(bedrooms.group(1)) if bedrooms else 0
    beds = int(beds.group(1)) if beds else 0

    return guests, bedrooms, beds
