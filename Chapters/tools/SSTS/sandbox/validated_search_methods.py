import pyrata.re as pyre

#Design rule based search mechanism
"""
Example of rule based search, defining properties of time series:

string: pppppppppppnnnnnnnnnnnnppppppppnnnnnnnnn
POS :            peak                peak
TYPE:            SYM                 SYM

"""

signal_str = "pppppppppppnnnnnnnnnnnppppppp"

#search for a peak, and save as a peak
ch_dict = [{"pos": "peak1", "raw":"pppppppppnnnnnnnnn", "s":"1"}, {"pos":"valley1", "raw":"nnnnnnnnnppppppp", "s":"0"}]

pattern = '[(pos="peak1"|pos~"valley.") & (s="0"|s="1")]'

print(pyre.findall(pattern, ch_dict))

