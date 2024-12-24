from scrapy import cmdline
import pandas as pd
import time
import sys
if len(sys.argv) > 2:
    building_name = sys.argv[1]
    id = sys.argv[2]
# print(building_name,id)

cmdline.execute(f'scrapy crawl wikipieda_spider -a building_name={building_name.replace(" ","")} -a id={str(id)}'.split())
