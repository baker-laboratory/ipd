import ipd

missing = ipd.lazyimport('does_not_exist')
missing.BOOM
