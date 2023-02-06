# MRZ Detection

## General Idea
### Get cropped image of the MRZ:
[This article](https://pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/)

### Use OCR to convert the MRZ into text:
MRZs use a specific font (OCR-B) and only 37 characters:
    
    ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<

On each line (of which, in a passport, there are 2), there are exactly 44 characters.

We need a dataset with 14325 occurrences of each character.

This should make it pretty easy to train an OCR on each line, returning a string of text.

### Parsing the MRZ
[The Specification](https://www.icao.int/publications/Documents/9303_p3_cons_en.pdf)


# Data Collection
- MIDV500 (38.9GB downloaded -> 2.37GB used)
- [PRADO](https://www.consilium.europa.eu/prado/en/search-by-document-title.html#letter1) has 2949 documents from all over the world. 
