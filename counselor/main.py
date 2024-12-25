from scrapy import cmdline
import pandas as pd
import time
import sys
if len(sys.argv) > 3:
    building_name = sys.argv[1]
    id = sys.argv[2]
    root = sys.argv[3]
# print(building_name,id)

cmdline.execute(f'scrapy crawl wikipieda_spider -a building_name={building_name.replace(" ","")} -a id={str(id)} -a root={str(root)}'.split())
