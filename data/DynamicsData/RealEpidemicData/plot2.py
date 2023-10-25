import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import h5py
import pycountry as pc
from fuzzywuzzy import fuzz
import geopandas as gp
import pickle
from itertools import product
from tqdm import tqdm_notebook as tqdm
import shapely

path_to_raw = "raw-dataset"
path_to_processed = "processed-dataset"

provincia_data = pd.read_csv(
    f"{path_to_raw}/cases-data.csv", delimiter=',', encoding="latin1", na_filter=False
)
prov_pop = pd.read_csv(
    f"{path_to_raw}/population-data.csv", error_bad_lines=False, encoding="latin1", delimiter=';'
)

subdiv = pc.subdivisions.get(country_code="ES")

days_and_months = {
    r"Jan. 2020": 31,
    r"Feb. 2020": 29,
    r"Mar. 2020": 31,
    r"Apr. 2020": 30,
    r"May 2020": 31,
    r"Jun. 2020": 30,
    r"Jul. 2020": 31,
    r"Aug. 2020": 31,
    r"Sep. 2020": 30,
    r"Oct. 2020": 31,
    r"Nov. 2020": 30,
    r"Dec. 2020": 31,
    r"Jan. 2021": 31,
    r"Feb. 2021": 28,
    r"Mar. 2021": 31
}
cumul_days = {}
c = 0
for m, d in days_and_months.items():
    cumul_days[m] = c
    c += d



# Names and ISO
names = []
i = 0
for x in provincia_data["provincia_iso"]:
    if x not in names:
        names.append(x)
name_map = {}
for n in names:
    for s in subdiv:
        if n == s.code[3:]:
            name_map[n] = s
            break
        elif s.code[3:] == "ML" and n == "ME":
            name_map[n] = s
            break

# Dates
dates = []
for x in provincia_data["fecha"]:
    if x not in dates:
        dates.append(x)

# Cases
cases = {}
for i in provincia_data["num_casos"].keys():
    prov = provincia_data["provincia_iso"][i]
    date = provincia_data["fecha"][i]
    num_cases = provincia_data["num_casos"][i]
    if prov in names:
        if prov not in cases:
            cases[prov] = {date: num_cases}
        elif date not in cases[prov]:
            cases[prov][date] = num_cases
        else:
            cases[prov][date] += num_cases



num_timesteps = len(dates)
num_nodes = 52

timeseries = np.zeros((num_timesteps, num_nodes))

index = 0
for n in names:
# for index in range(num_nodes):
    if n == 'NC':
        continue
    # cases_labeled[n] = np.zeros(num_timesteps)
    for time in range(num_timesteps):
        d = dates[time]
        timeseries[time, index] = cases[n][d]
    index += 1
#
# path = f"{path_to_processed}/case-timeseries.data"
# with open(path, "wb") as f:
#     pickle.dump(timeseries, f)
#
# timeseries = pickle.load(open(path, "rb"))

# from scipy.stats import norm
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def ridgeline(data, x=None, overlap=0, yshift=0., color="blue", zorder=0):
#     curves = []
#     ys = []
#     labels = []
#     for i, (k, v) in enumerate(data.items()):
#         labels.append(k)
#         d = v / v.max() * (1. + overlap)
#         if x is None:
#             x = np.arange(len(d))
#         y = i  # *(1.0 - overlap)
#         ys.append(y)
#         plt.fill_between(x, np.ones(len(d)) * y + yshift, d + y + yshift, zorder=zorder + len(data) - i + 1,
#                          color=color, alpha=0.5)
#         plt.plot(x, d + y, c='k', zorder=len(data) - i + 1)
#     plt.yticks(ys, labels, fontsize=10)
#
#
# ax = plt.gca()
# ax.plot(timeseries.sum(1))
#
# ax.set_xlabel('Time', fontsize=14)
# ax.set_ylabel('Incidence', fontsize=14)
# plt.show()
#
# plt.figure(figsize=(12, 8))
# ax = plt.gca()
# ridgeline(cases_labeled, yshift=0, overlap=0., color="blue", zorder=0)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)
#
# months = list(cumul_days.keys())
# ax.set_xticks(list(cumul_days.values()))
# ax.set_xticklabels(months, rotation=45, fontsize=12)
# plt.show()



prov_pop = pd.read_csv(
    f"{path_to_raw}/population-data.csv", error_bad_lines=False, encoding="latin1", delimiter=';'
)


# Provinces
pop = {}
count = 0
for n in name_map:
    for i, p in enumerate(prov_pop["Provincias"]):
        # print(i, p)
        # print(fuzz.token_set_ratio(name_map[n].name, p))
        # print(fuzz.token_set_ratio(name_map[n].name, p), name_map[n].name, p, n, count)
        # if fuzz.token_set_ratio(name_map[n].name, p) > 80:
        if fuzz.token_set_ratio(name_map[n].name, p) >= 80:
            count += 1
            if n == 'NA':
                print('--NA')
            print(n, name_map[n].name, p, count, name_map[n].code[3:])
            if n not in pop:
                pop[n] = int(prov_pop["Total"][i].replace(".", ""))
            elif n in pop and name_map[n].parent_code is not None:
                pop[n] = int(prov_pop["Total"][i].replace(".", ""))
pop['NA'] = int(prov_pop["Total"][32].replace(".", ""))
# pop['NC'] = int(prov_pop["Total"][32].replace(".", ""))
# assert len(pop) == len(names), f"{len(pop)} != {len(names)}"


# {'NC', 'NA', 'L', 'GI'}
# - 67 80 80


# num_nodes = 52
# population = np.zeros(num_nodes)
# index = 0
# for n in names:
#     if n == 'NC':
#         continue
#     population[index] = pop[n]
#     index += 1

# path = f"{path_to_processed}/population.data"
# with open(path, "wb") as f:
#     pickle.dump(population, f)

# exit(1)
ine_to_names = {
    1: "Araba/Álava",
    2: "Albacete",
    3: "Alicante/Alacant",
    4: "Almería",
    5: "Ávila",
    6: "Badajoz",
    7: "Balears",
    8: "Barcelona",
    9: "Burgos",
    10: "Cáceres",
    11: "Cádiz",
    12: "Castellón/Castelló",
    13: "Ciudad Real",
    14: "Córdoba",
    15: "Coruña, A",
    16: "Cuenca",
    17: "Girona",
    18: "Granada",
    19: "Guadalajara",
    20: "Gipuzkoa",
    21: "Huelva",
    22: "Huesca",
    23: "Jaén",
    24: "León",
    25: "Lleida",
    26: "Rioja, La",
    27: "Lugo",
    28: "Madrid",
    29: "Málaga",
    30: "Murcia",
    31: "Navarra",
    32: "Ourense",
    34: "Palencia",
    33: "Asturias",
    35: "Palmas, Las",
    36: "Pontevedra",
    37: "Salamanca",
    38: "Santa Cruz de Tenerife",
    39: "Cantabria",
    40: "Segovia",
    41: "Sevilla",
    42: "Soria",
    43: "Tarragona",
    44: "Teruel",
    45: "Toledo",
    46: "Valencia/València",
    47: "Valladolid",
    48: "Bizkaia",
    49: "Zamora",
    50: "Zaragoza",
    51: "Ceuta",
    52: "Melilla"
}
names_to_ine = {v: k for k, v in ine_to_names.items()}

ine_to_iso_map = {}
for i, n in ine_to_names.items():
    for s in subdiv:
        if fuzz.token_set_ratio(n, s.name) >= 80:
            if i not in ine_to_iso_map:
                ine_to_iso_map[i] = s.code[3:]
            elif i in ine_to_iso_map and s.parent_code is not None:
                ine_to_iso_map[i] = s.code[3:]
                break
ine_to_iso_map[52] = "ME"

#                 ine_to_ccaa_map[k] = s.parent_code[3:]
#             else:
#                 ine_to_ccaa_map[k] = "None"
#             break
iso_to_ine_map = {v: k for k, v in ine_to_iso_map.items()}

# ccaa_to_ine_map = {}
# i = 0
# for k, v in ine_to_ccaa_map.items():
#     if v == "None":
#         v = f"None{i}"
#         i += 1
#     if v not in ccaa_to_ine_map:
#         ccaa_to_ine_map[v] = [k]
#     else:
#         ccaa_to_ine_map[v].append(k)


import zipfile

all_files = {}

# Loading july data
path_to_unzipped = f"{path_to_raw}/matrices_julio_csv"
with zipfile.ZipFile(f"{path_to_unzipped}.zip", 'r') as zip_ref:
    zip_ref.extractall(f"{path_to_unzipped}")
prefix = f"{path_to_unzipped}/matrices_julio/VI_J"
suffix = ".csv"
for i in range(13):
    all_files[f"J{i}"] = prefix + str(i).zfill(2) + suffix

# Loading october data
path_to_unzipped = f"{path_to_raw}/matrices_octubre_csv"
with zipfile.ZipFile(f"{path_to_unzipped}.zip", 'r') as zip_ref:
    zip_ref.extractall(f"{path_to_unzipped}")
prefix = f"{path_to_unzipped}/matrices_octubre/VI_O"
for i in range(16):
    all_files[f"O{i}"] = prefix + str(i).zfill(2) + suffix


def get_mobility_matrix(csvfile):
    prov_m = {
        "all": np.zeros((len(ine_to_iso_map), len(ine_to_iso_map))),
        "plane": np.zeros((len(ine_to_iso_map), len(ine_to_iso_map))),
        "car": np.zeros((len(ine_to_iso_map), len(ine_to_iso_map))),
        "bus": np.zeros((len(ine_to_iso_map), len(ine_to_iso_map))),
        "boat": np.zeros((len(ine_to_iso_map), len(ine_to_iso_map))),
        "train": np.zeros((len(ine_to_iso_map), len(ine_to_iso_map))),
    }
    # ccaa_m = {
    #     "all": np.zeros((len(ccaa_to_ine_map), len(ccaa_to_ine_map))),
    #     "plane": np.zeros((len(ccaa_to_ine_map), len(ccaa_to_ine_map))),
    #     "car": np.zeros((len(ccaa_to_ine_map), len(ccaa_to_ine_map))),
    #     "bus": np.zeros((len(ccaa_to_ine_map), len(ccaa_to_ine_map))),
    #     "boat": np.zeros((len(ccaa_to_ine_map), len(ccaa_to_ine_map))),
    #     "train": np.zeros((len(ccaa_to_ine_map), len(ccaa_to_ine_map))),
    # }

    for index in csvfile["Origen"].keys():
        if type(csvfile["Origen"][index]) == int:
            i = csvfile["Origen"][index]
        elif len(csvfile["Origen"][index]) == 2:
            i = int(csvfile["Origen"][index])
        elif len(csvfile["Origen"][index]) >= 2:
            i = int(csvfile["Origen"][index][:2])

        if type(csvfile["Destino"][index]) == int:
            i = csvfile["Destino"][index]
        elif len(csvfile["Destino"][index]) == 2:
            j = int(csvfile["Destino"][index])
        elif len(csvfile["Destino"][index]) >= 2:
            j = int(csvfile["Destino"][index][:2])

        w = float(csvfile["Viajeros"][index].replace(',', ''))

        prov_m["all"][int(i - 1), int(j - 1)] += w
        if csvfile["Modo"][index] == "privado" or csvfile["Modo"][index] == "carretera":
            prov_m["car"][int(i - 1), int(j - 1)] += w
        if csvfile["Modo"][index] == "autobús":
            prov_m["bus"][int(i - 1), int(j - 1)] += w
        elif csvfile["Modo"][index] == "tren":
            prov_m["train"][int(i - 1), int(j - 1)] += w
        elif csvfile["Modo"][index] == "avión":
            prov_m["plane"][int(i - 1), int(j - 1)] += w
        elif csvfile["Modo"][index] == "barco":
            prov_m["boat"][int(i - 1), int(j - 1)] += w

    # for (i, m), (j, n) in product(enumerate(ccaa_to_ine_map), enumerate(ccaa_to_ine_map)):
    #     for k, l in product(ccaa_to_ine_map[m], ccaa_to_ine_map[n]):
    #         k -= 1
    #         l -= 1
    #         ccaa_m["all"][i, j] += prov_m["all"][k, l]
    #         ccaa_m["car"][i, j] += prov_m["car"][k, l]
    #         ccaa_m["bus"][i, j] += prov_m["bus"][k, l]
    #         ccaa_m["plane"][i, j] += prov_m["plane"][k, l]
    #         ccaa_m["train"][i, j] += prov_m["train"][k, l]
    #         ccaa_m["boat"][i, j] += prov_m["boat"][k, l]
    ccaa_m =None
    return prov_m, ccaa_m

prov_mobility = {}
ccaa_mobility = {}
for k, v in all_files.items():
    print(k)
    prov_m, ccaa_m = get_mobility_matrix(pd.read_csv(v))
    for kk in prov_m.keys():
        if kk not in prov_mobility:
            prov_mobility[kk] = prov_m[kk] / len(all_files)
            # ccaa_mobility[kk] = ccaa_m[kk] / len(all_files)
        else:
            prov_mobility[kk] += prov_m[kk] / len(all_files)
            # ccaa_mobility[kk] += ccaa_m[kk] / len(all_files)

prov_pop = {k: pop[v] if v != "ME" else pop["ML"] for k, v in ine_to_iso_map.items()}


print(list(ine_to_iso_map.values()))
# save province_mobility

path = f"{path_to_processed}/province_mobility.data"
with open(path, "wb") as f:
    pickle.dump(prov_mobility, f)

# save population
population = np.zeros(num_nodes)

for k, v in prov_pop.items():
    population[k-1] = v

path = f"{path_to_processed}/population.data"
with open(path, "wb") as f:
    pickle.dump(population, f)

# save timeseries
ine_to_iso_map_for_timeseries = {}
names.remove('NC')
for k, v in ine_to_iso_map.items():
    for ii in range(len(names)):
        n = names[ii]
        if n == 'ML':
            n = 'ME'

        if n == v:
            ine_to_iso_map_for_timeseries[k] = ii

timeseries_rearrange = np.zeros_like(timeseries)
for i in range(len(list(ine_to_iso_map_for_timeseries.values()))):
    timeseries_rearrange[:,i] = timeseries[:, list(ine_to_iso_map_for_timeseries.values())[i]]

path = f"{path_to_processed}/case-timeseries.data"
with open(path, "wb") as f:
    pickle.dump(timeseries_rearrange, f)



exit(1)

from scipy.optimize import bisect

def thresholding(m, t):
    m = np.log(m + 1)
    m = 0.5 * (m + m.T)
    x = np.zeros(m.shape)
    t *= np.max(m)
    x[m > t] = 1.
    return x

def select_threshold(m, avgk):
    f_to_solve = lambda t: thresholding(m, t).sum(0).mean() - avgk
    return bisect(f_to_solve, 0.01, 0.9)

avgk = 10
thresh = select_threshold(prov_mobility["all"], avgk)
prov_mobility["thresholded"] = thresholding(prov_mobility["all"], thresh)

prov_mobility
g_dict = {}

for k, v in prov_mobility.items():
    g = nx.DiGraph()
    for i, j in product(range(v.shape[0]), range(v.shape[1])):
        w = v[i, j]
        if w > 0:
            g.add_edge(i, j, weight=v[i, j])


edge_list = {k: np.array(np.where(v > 0)).T for k, v in prov_mobility.items()}
node_list = {k: np.arange(v.shape[0]) for k, v in prov_mobility.items()}
edge_attr = {
    "weight":{
        k: np.array([prov_mobility[k][u, v] for u,v in edge_list[k]]) for k, v in prov_mobility.items()
    }
}
node_attr = {"population": {k: timeseries.sum(-1).mean(0) for k, v in node_list.items()}}
X = timeseries / np.expand_dims(timeseries.sum(-1).mean(0), -1)


with h5py.File(f"{path_to_processed}/spain-covid19-dataset.h5", "w") as f:

    group = f.create_group("thresholded")
    group = group.create_group("data")
    if "timeseries" in group.keys():
        del group["timeseries"]

    if "networks" in group.keys():
        del group["networks"]
    group.create_dataset("timeseries/d0", data=X)
    group.create_dataset("networks/d0/edge_list", data=edge_list["thresholded"])
    group.create_dataset("networks/d0/node_list", data=node_list["thresholded"])
    group.create_dataset("networks/d0/node_attr/population", data=node_attr["population"]["thresholded"])

    group = f.create_group("weighted")
    group = group.create_group("data")
    if "timeseries" in group.keys():
        del group["timeseries"]

    if "networks" in group.keys():
        del group["networks"]
    group.create_dataset("timeseries/d0", data=X)
    group.create_dataset("networks/d0/edge_list", data=edge_list["all"])
    group.create_dataset("networks/d0/node_list", data=node_list["all"])
    group.create_dataset("networks/d0/edge_attr/weight", data=edge_attr["weight"]["all"])
    group.create_dataset("networks/d0/node_attr/population", data=node_attr["population"]["all"])

    group = f.create_group("multiplex")
    group = group.create_group("data")
    if "timeseries" in group.keys():
        del group["timeseries"]

    if "networks" in group:
        del group["networks"]
    group.create_dataset("timeseries/d0", data=X)
    for k in edge_list.keys():
        group.create_dataset(f"networks/d0/{k}/edge_list", data=edge_list[k])
        group.create_dataset(f"networks/d0/{k}/node_list", data=node_list[k])
        group.create_dataset(f"networks/d0/{k}/node_attr/population", data=node_attr["population"][k])

    group = f.create_group("weighted-multiplex")
    group = group.create_group("data")
    if "timeseries" in group.keys():
        del group["timeseries"]

    if "networks" in group:
        del group["networks"]
    group.create_dataset("timeseries/d0", data=X)
    for k in edge_list.keys():

        if k != "thresholded" and k != "all":
            group.create_dataset(f"networks/d0/{k}/edge_list", data=edge_list[k])
            group.create_dataset(f"networks/d0/{k}/node_list", data=node_list[k])
            group.create_dataset(f"networks/d0/{k}/edge_attr/weight", data=edge_attr["weight"][k])
            group.create_dataset(f"networks/d0/{k}/node_attr/population", data=node_attr["population"][k])

