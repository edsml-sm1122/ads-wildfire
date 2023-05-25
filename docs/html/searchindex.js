Search.setIndex({"docnames": ["api/wildfire.ConvLSTM", "api/wildfire.ConvLSTMCell", "api/wildfire.KalmanGain", "api/wildfire.VAE_Conv", "api/wildfire.VAE_Decoder_Conv", "api/wildfire.VAE_Encoder_Conv", "api/wildfire.covariance_matrix", "api/wildfire.create_dataloader", "api/wildfire.create_pairs", "api/wildfire.run_assimilation", "api/wildfire.split", "api/wildfire.train", "api/wildfire.update_prediction", "index"], "filenames": ["api/wildfire.ConvLSTM.rst", "api/wildfire.ConvLSTMCell.rst", "api/wildfire.KalmanGain.rst", "api/wildfire.VAE_Conv.rst", "api/wildfire.VAE_Decoder_Conv.rst", "api/wildfire.VAE_Encoder_Conv.rst", "api/wildfire.covariance_matrix.rst", "api/wildfire.create_dataloader.rst", "api/wildfire.create_pairs.rst", "api/wildfire.run_assimilation.rst", "api/wildfire.split.rst", "api/wildfire.train.rst", "api/wildfire.update_prediction.rst", "index.rst"], "titles": ["ConvLSTM", "ConvLSTMCell", "KalmanGain", "VAE_Conv", "VAE_Decoder_Conv", "VAE_Encoder_Conv", "covariance_matrix", "create_dataloader", "create_pairs", "run_assimilation", "split", "train", "update_prediction", "Welcome to ads-wildfire\u2019s documentation!"], "terms": {"index": 13, "modul": [0, 1, 3, 4, 5, 11, 13], "search": 13, "page": 13, "class": [0, 1, 3, 4, 5], "vae_conv": [], "devic": [3, 11], "combin": 3, "convolut": [0, 1, 3, 4, 5], "encod": [3, 5], "decod": [3, 4], "vae": 3, "latent": [3, 4, 5], "space": [3, 4, 5], "paramet": [], "str": [3, 7, 11], "which": 3, "perform": [0, 1, 3], "comput": [0, 3], "vae_encoder_conv": [], "network": [], "vae_decoder_conv": [], "distribut": [], "torch": [0, 1, 7, 11], "normal": [], "us": [11, 12], "sampl": 3, "from": [3, 7, 8], "forward": [0, 1, 3, 4, 5], "x": [3, 4, 5, 6, 8, 12], "pass": [0, 1, 3, 4, 5], "float": [3, 4, 5], "batch": [1, 3], "imag": [1, 3, 4, 5], "data": [3, 6, 7, 8, 11], "loader": 3, "return": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "reconstruct": [3, 4], "kl_div": 3, "kl": 3, "diverg": 3, "term": 3, "regular": 3, "type": [], "z": 3, "sample_latent_spac": 3, "mu": [3, 5], "sigma": [3, 5], "mean": [3, 5], "learn": [3, 5], "represent": [4, 5], "standard": 3, "deviat": 3, "vector": [3, 12], "contain": [0, 1, 8], "layer": 0, "layer1": [], "convtranspose2d": [], "120": [], "input": [0, 1, 6, 8, 10], "channel": [0, 1], "60": [], "output": [0, 8], "kernel": [0, 1], "size": [0, 1, 8, 10], "4": [], "stride": [], "2": [], "pad": [], "1": [0, 9, 12], "follow": [], "gelu": [], "activ": [], "layer2": [], "40": [], "layer3": [], "20": [], "layer4": [], "10": 11, "layer5": [], "5": [], "tanh": [], "conv2d": [], "maxpool2d": [], "3": 11, "layermu": [], "layersigma": [], "stdev": 5, "convlstm": 1, "input_dim": [0, 1], "hidden_dim": [0, 1], "kernel_s": [0, 1], "num_lay": 0, "batch_first": 0, "fals": 0, "bia": [0, 1], "true": 0, "return_all_lay": 0, "lstm": [0, 1], "initi": [0, 1], "int": [0, 1, 8, 10, 11], "number": [0, 1, 11], "list": [0, 9, 10], "hidden": [0, 1], "tupl": [0, 1, 8], "stack": 0, "each": [0, 8, 9, 10], "other": 0, "bool": [0, 1], "whether": [0, 1], "dimens": 0, "0": 0, "i": 0, "b": [0, 2], "all": 0, "input_tensor": [0, 1], "hidden_st": 0, "none": 0, "tensor": [0, 1], "shape": [0, 1, 2, 6, 9, 12], "t": 0, "c": 0, "h": [0, 2, 12], "w": 0, "cell": [0, 1], "state": [0, 1, 9, 12], "two": [0, 8], "length": 0, "layer_output_list": 0, "last_state_list": 0, "last": 0, "convlstmcel": [], "add": 1, "cur_stat": 1, "batch_siz": [1, 7], "height": 1, "width": 1, "current": [1, 12], "next": 1, "init_hidden": 1, "image_s": 1, "dataload": [7, 11], "dataset": [], "t_co": [], "option": [7, 11], "shuffl": [], "sampler": [], "union": [], "iter": [], "batch_sampl": [], "sequenc": [], "num_work": [], "collate_fn": [], "callabl": [], "ani": [], "pin_memori": [], "drop_last": [], "timeout": [], "worker_init_fn": [], "multiprocessing_context": [], "gener": [], "prefetch_factor": [], "persistent_work": [], "pin_memory_devic": [], "provid": [], "an": 10, "over": [], "given": [], "The": [7, 8, 10, 11], "support": [], "both": [], "map": [], "style": [], "singl": [], "multi": [], "process": 9, "load": [], "custom": [], "order": [], "automat": [], "collat": [], "memori": [], "pin": [], "see": [], "util": [7, 11], "more": [], "detail": [], "how": [], "mani": [], "per": [], "default": [7, 11], "set": [], "have": [], "reshuffl": [], "everi": [], "epoch": 11, "defin": [], "strategi": [], "draw": [], "can": [], "__len__": [], "implement": [], "If": [], "specifi": [], "must": [], "like": [], "time": [], "mutual": [], "exclus": [], "subprocess": [], "main": [], "merg": [], "form": [], "mini": [], "when": [], "copi": [], "cuda": [], "befor": 11, "them": [], "your": [], "element": [], "ar": [], "exampl": [], "below": [], "drop": [], "incomplet": [], "divis": [], "smaller": [], "numer": [], "posit": [], "valu": 9, "collect": [], "worker": [], "should": [], "alwai": [], "non": [], "neg": [], "thi": [], "call": [], "id": [], "after": [], "seed": [], "rng": [], "randomsampl": [], "random": [], "multiprocess": [], "base_se": [], "keyword": [], "onli": [], "arg": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "advanc": [], "total": [], "prefetch": [], "across": [], "depend": [], "otherwis": [], "shutdown": [], "ha": [], "been": [], "consum": [], "onc": [], "allow": [], "maintain": [], "instanc": [], "aliv": [], "spawn": [], "start": [], "method": [0, 1, 3, 4, 5], "cannot": [], "unpickl": [], "object": [], "e": [], "g": [], "lambda": [], "function": [], "best": [], "practic": [], "relat": [], "pytorch": [], "len": [], "heurist": [], "base": [0, 1, 3, 4, 5], "iterabledataset": [], "instead": [], "estim": [9, 12], "proper": [], "round": [], "regardless": [], "configur": [], "repres": [], "guess": [], "make": [], "becaus": [], "trust": [], "user": [], "code": [], "correctli": [], "handl": [], "avoid": [], "duplic": [], "howev": [], "shard": [], "result": [], "multipl": 10, "still": [], "inaccur": [], "complet": [], "broken": [], "ones": [], "than": [], "one": [], "worth": [], "unfortun": [], "detect": [], "case": [], "_": [], "interact": [], "reproduc": [], "note": [], "question": [], "kalmangain": [], "r": 2, "calcul": [2, 6], "kalman": [2, 12], "gain": [2, 12], "matrix": [2, 6, 9, 12], "numpi": [2, 6, 8, 9, 10, 12], "ndarrai": [2, 6, 8, 9, 10, 12], "background": 2, "error": 2, "covari": [2, 6], "n": [2, 6, 9, 12], "measur": [2, 12], "p": [2, 9, 12], "plotloss": [], "bo": [], "matplotlibplot": [], "extremaprint": [], "mode": 7, "notebook": [], "kwarg": [], "metric": [], "train": 7, "engin": [], "send": [], "plugin": [], "inherit": [], "baseoutput": [], "string": [], "livelossplot": [], "built": [], "script": [], "some": [], "need": [], "chang": [], "behavior": [], "work": [], "environ": [], "kei": [], "argument": [], "mainlogg": [], "constructor": [], "substitut": [], "old": [], "api": [], "reset_output": [], "reset": [], "so": [], "chain": [], "log": [], "to_bokeh": [], "append": [], "bokehplot": [], "to_extrema_print": [], "to_matplotlib": [], "to_neptun": [], "neptunelogg": [], "to_tensorboard": [], "tensorboardlogg": [], "to_tensorboard_tf": [], "tensorboardtflogg": [], "updat": [9, 12], "logger": [], "tensordataset": [], "wrap": [], "retriev": [], "along": [], "first": [], "same": [], "initilis": 3, "initialis": [4, 5], "covariance_matrix": [], "m": [6, 9], "create_dataload": [], "path": 7, "creat": [7, 8], "split": [7, 8], "file": 7, "create_pair": [], "chunk_siz": [8, 10], "pair": 8, "A": 8, "arrai": [8, 10], "y": [8, 12], "inv": [], "invers": [], "squar": [], "ainv": [], "satisfi": [], "dot": [], "ey": [], "array_lik": [], "invert": [], "rais": [], "linalgerror": [], "fail": [], "scipi": [], "linalg": [], "similar": [], "new": [], "version": [], "8": [], "broadcast": [], "rule": [], "appli": [], "import": [], "np": [], "allclos": [], "well": [], "sever": [], "matric": [], "25": [], "75": [], "run_assimil": [], "preds_compr": 9, "obs_data_compr": 9, "run": 9, "assimil": 9, "predict": [9, 12], "observ": 9, "arr": 10, "subarrai": 10, "equal": 10, "tqdm": [], "__": [], "decor": [], "act": [], "exactli": [], "origin": [], "print": [], "dynam": [], "progressbar": [], "request": [], "leav": [], "blank": [], "manual": [], "manag": [], "desc": [], "prefix": [], "expect": [], "unspecifi": [], "possibl": [], "inf": [], "resort": [], "basic": [], "progress": [], "statist": [], "displai": [], "eta": [], "gui": [], "subsequ": [], "arbitrari": [], "larg": [], "9e9": [], "keep": [], "trace": [], "upon": [], "termin": [], "io": [], "textiowrapp": [], "stringio": [], "where": [], "messag": [], "sy": [], "stderr": [], "write": [], "flush": [], "For": [], "write_byt": [], "ncol": [], "entir": [], "resiz": [], "stai": [], "within": [], "bound": [], "attempt": [], "fallback": [], "meter": [], "limit": [], "counter": [], "stat": [], "mininterv": [], "minimum": [], "interv": [], "second": [], "maxinterv": [], "maximum": [], "adjust": [], "minit": [], "correspond": [], "long": [], "lag": [], "dynamic_minit": [], "monitor": [], "thread": [], "enabl": [], "cpu": 11, "effici": [], "good": [], "tight": [], "loop": [], "skip": [], "tweak": [], "get": [], "veri": [], "errat": [], "fast": [], "slow": [], "item": [], "etc": [], "you": [], "ascii": [], "unicod": [], "smooth": [], "block": [], "fill": [], "charact": [], "123456789": [], "disabl": [], "wrapper": [], "tty": [], "unit": [], "unit_scal": [], "reduc": [], "scale": [], "intern": [], "system": [], "kilo": [], "mega": [], "zero": [], "dynamic_ncol": [], "constantli": [], "alter": [], "nrow": [], "window": [], "exponenti": [], "move": [], "averag": [], "factor": [], "speed": [], "ignor": [], "rang": [], "instantan": [], "bar_format": [], "bar": [], "format": [], "mai": [], "impact": [], "l_bar": [], "r_bar": [], "percentag": [], "0f": [], "n_fmt": [], "total_fmt": [], "elaps": [], "remain": [], "rate_fmt": [], "postfix": [], "var": [], "elapsed_": [], "rate": [], "rate_noinv": [], "rate_noinv_fmt": [], "rate_inv": [], "rate_inv_fmt": [], "unit_divisor": [], "remaining_": [], "trail": [], "remov": [], "latter": [], "empti": [], "restart": [], "consid": [], "3f": [], "line": [], "offset": [], "eg": [], "dict": [], "addit": [], "end": [], "set_postfix": [], "1000": [], "unless": [], "byte": [], "written": [], "python": [], "also": [], "In": [], "lock_arg": [], "refresh": [], "intermedi": [], "screen": [], "hide": [], "nest": [], "outsid": [], "colour": [], "green": [], "00ff00": [], "delai": [], "don": [], "until": [], "warn": [], "do": [], "matplotlib": [], "anim": [], "graphic": [], "out": [], "clear": [], "nolock": [], "close": [], "cleanup": [], "msg": [], "po": [], "self": [], "sp": [], "overload": [], "some_frontend": [], "format_dict": [], "what": [], "repr": [], "moveto": [], "ab": [], "classmethod": [], "external_write_mod": [], "context": [], "exit": [], "stream": [], "properti": [], "public": [], "read": [], "member": [], "access": [], "static": [], "format_interv": [], "clock": [], "mm": [], "ss": [], "format_met": [], "extra_kwarg": [], "finish": [], "meaningless": [], "sinc": [], "includ": [], "appropri": [], "si": [], "k": 12, "6": [], "overrid": [], "place": [], "usual": [], "382": [], "readi": [], "format_num": [], "intellig": [], "scientif": [], "notat": [], "3g": [], "format_sizeof": [], "num": [], "suffix": [], "divisor": [], "greater": [], "uniti": [], "magnitud": [], "post": [], "between": [], "get_lock": [], "global": [], "lock": [], "construct": [], "doe": [], "exist": [], "panda": [], "tqdm_kwarg": [], "regist": [], "core": [], "frame": [], "datafram": [], "seri": [], "groupbi": [], "dataframegroupbi": [], "seriesgroupbi": [], "progress_appli": [], "pd": [], "tqdm_gui": [], "df": [], "randint": [], "100": [], "100000": [], "50": [], "now": [], "refer": [], "http": [], "stackoverflow": [], "com": [], "18603270": [], "dure": [], "oper": [], "forc": [], "acquir": [], "repeat": [], "set_descript": [], "modifi": [], "descript": [], "set_description_str": [], "without": [], "set_lock": [], "ordered_dict": [], "datatyp": [], "ordereddict": [], "set_postfix_str": [], "dictionari": [], "expans": [], "status_print": [], "longer": [], "unpaus": [], "timer": [], "files": [], "current_buff": [], "highli": [], "recommend": [], "possibli": [], "necessari": [], "wai": [], "reach": [], "increment": [], "wa": [], "trigger": [], "wrapattr": [], "file_obj": [], "fobj": [], "while": [], "chunk": [], "break": [], "via": [], "overlap": [], "model": 11, "train_data": 11, "val_data": 11, "patienc": 11, "nn": 11, "valid": 11, "wait": 11, "improv": 11, "loss": 11, "earli": 11, "stop": 11, "autoencod": 11, "update_predict": [], "filter": 12, "equat": 12, "wildfir": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "summari": [0, 1, 3, 4, 5], "document": [0, 1, 3, 4, 5], "clstm": [], "cvae": []}, "objects": {"": [[13, 0, 0, "-", "wildfire"]], "wildfire": [[0, 1, 1, "", "ConvLSTM"], [1, 1, 1, "", "ConvLSTMCell"], [2, 3, 1, "", "KalmanGain"], [3, 1, 1, "", "VAE_Conv"], [4, 1, 1, "", "VAE_Decoder_Conv"], [5, 1, 1, "", "VAE_Encoder_Conv"], [6, 3, 1, "", "covariance_matrix"], [7, 3, 1, "", "create_dataloader"], [8, 3, 1, "", "create_pairs"], [9, 3, 1, "", "run_assimilation"], [10, 3, 1, "", "split"], [11, 3, 1, "", "train"], [12, 3, 1, "", "update_prediction"]], "wildfire.ConvLSTM": [[0, 2, 1, "", "forward"]], "wildfire.ConvLSTMCell": [[1, 2, 1, "", "forward"], [1, 2, 1, "", "init_hidden"]], "wildfire.VAE_Conv": [[3, 2, 1, "", "forward"], [3, 2, 1, "", "sample_latent_space"]], "wildfire.VAE_Decoder_Conv": [[4, 2, 1, "", "forward"]], "wildfire.VAE_Encoder_Conv": [[5, 2, 1, "", "forward"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "function", "Python function"]}, "titleterms": {"welcom": 13, "ad": 13, "wildfir": 13, "": 13, "document": 13, "indic": 13, "tabl": 13, "convlstm": 0, "convlstmcel": 1, "kalmangain": 2, "vae_conv": 3, "vae_decoder_conv": 4, "vae_encoder_conv": 5, "covariance_matrix": 6, "create_dataload": 7, "create_pair": 8, "run_assimil": 9, "split": 10, "train": 11, "update_predict": 12, "packag": 13, "function": 13, "class": 13, "inherit": [], "diagram": [], "util": [], "modul": []}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 57}, "alltitles": {"KalmanGain": [[2, "kalmangain"]], "VAE_Conv": [[3, "vae-conv"]], "VAE_Decoder_Conv": [[4, "vae-decoder-conv"]], "VAE_Encoder_Conv": [[5, "vae-encoder-conv"]], "covariance_matrix": [[6, "covariance-matrix"]], "create_dataloader": [[7, "create-dataloader"]], "create_pairs": [[8, "create-pairs"]], "run_assimilation": [[9, "run-assimilation"]], "split": [[10, "split"]], "train": [[11, "train"]], "update_prediction": [[12, "update-prediction"]], "Welcome to ads-wildfire\u2019s documentation!": [[13, "welcome-to-ads-wildfire-s-documentation"]], "Indices and tables": [[13, "indices-and-tables"]], "wildfire Package": [[13, "module-wildfire"]], "Functions": [[13, "functions"]], "Classes": [[13, "classes"]], "ConvLSTM": [[0, "convlstm"]], "ConvLSTMCell": [[1, "convlstmcell"]]}, "indexentries": {"convlstm (class in wildfire)": [[0, "wildfire.ConvLSTM"]], "forward() (wildfire.convlstm method)": [[0, "wildfire.ConvLSTM.forward"]], "convlstmcell (class in wildfire)": [[1, "wildfire.ConvLSTMCell"]], "forward() (wildfire.convlstmcell method)": [[1, "wildfire.ConvLSTMCell.forward"]], "init_hidden() (wildfire.convlstmcell method)": [[1, "wildfire.ConvLSTMCell.init_hidden"]], "module": [[13, "module-wildfire"]], "wildfire": [[13, "module-wildfire"]]}})