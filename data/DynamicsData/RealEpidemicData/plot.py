import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import geopandas

# def preprocessing_real_epidemic_data(disease_type='H1N1'):
#
#     if disease_type == 'H1N1':
#         infected_numbers = pd.read_csv('raw_data/infected_numbers_H1N1.csv', encoding='utf-8', header=None)
#         timepoints = infected_numbers.values[:, 0]
#         infected_numbers = infected_numbers.values[:, 1:]
#     elif disease_type == 'SARS':
#         infected_numbers = pd.read_csv('raw_data/infected_numbers_sars.csv', encoding='utf-8', header=None)
#         timepoints = np.array(infected_numbers.values[1:, 0],dtype=np.int64)
#         infected_numbers = np.array(infected_numbers.values[1:, 1:],dtype=np.int64)
#     elif disease_type == 'COVID':
#         infected_numbers = pd.read_csv('raw_data/infected_numbers_covid.csv', encoding='utf-8', header=None)
#         timepoints = np.array(infected_numbers.values[1:, 0],dtype=np.int64)
#         infected_numbers = np.array(infected_numbers.values[1:, 1:],dtype=np.int64)
#     else:
#         print('ERROR disease_type [%s]'%disease_type)
#         exit(1)
#
#     A = pd.read_csv('raw_data/Flights_adj.csv', encoding='utf-8', header=None)
#     A = A.values
#
#     populations = pd.read_csv('raw_data/populations.csv', encoding='utf-8', header=None)
#     populations = populations.values.reshape(-1)
#     countries = pd.read_csv('raw_data/Country_Population_final.csv', encoding='utf-8', header=None)
#     countries = countries.values[1:, 1]
#
#     with_infected = np.mean(infected_numbers, axis=0) > 0
#     populations = populations[with_infected]
#     countries = countries[with_infected]
#     A = A[with_infected, :][:, with_infected]
#     infected_numbers = infected_numbers[:, with_infected]
#
#     x = np.concatenate([timepoints.reshape(-1, 1), infected_numbers], axis=-1)
#     fully_timepoints = np.linspace(0, x[-1, 0], x[-1, 0] + 1, dtype=np.int64)
#     missing_timepoints = np.array(list(set(fully_timepoints.tolist()) - set(timepoints.tolist())))
#
#     # interp missing
#     interp_points = np.zeros((len(missing_timepoints), infected_numbers.shape[-1]), dtype=np.int64)
#     for n in range(infected_numbers.shape[-1]):
#         for t in range(len(missing_timepoints)):
#             interp_points[t, n] = np.ceil(np.interp(missing_timepoints[t], timepoints, infected_numbers[:, n]))
#
#     interp_points = np.concatenate([missing_timepoints.reshape(-1, 1), interp_points], axis=-1)
#     x = np.concatenate([x, interp_points], axis=0)
#
#     sorted_indexs = np.argsort(x[:, 0])
#     x = x[sorted_indexs, :]
#
#     x_values = x[:, 1:]  # without time columns
#
#     self_air = np.diag(A)
#     city_size = self_air / np.sum(self_air) * populations
#     norm_x = x_values * np.repeat(city_size.reshape(1, -1), x_values.shape[0], axis=0)
#     Aij = A - np.diag(np.diag(A))
#     travel_person = populations * (8.91e6 / np.sum(populations))
#     norm_xj = x_values / np.repeat(travel_person.reshape(1, -1), x_values.shape[0], axis=0)
#     city_activity = 8.91e6 / np.sum(populations)
#     Aij_act = Aij * city_activity
#
#     np.savetxt('%s_adj.txt'%disease_type, Aij_act)
#     np.savetxt('%s_filter_total_cases_raw.txt'%disease_type, x_values)
#     np.savetxt('%s_node_name_idx.txt'%disease_type, np.linspace(0, len(with_infected)-1, len(with_infected))[with_infected])
#
#
#
# preprocessing_real_epidemic_data(disease_type='H1N1')
#
# preprocessing_real_epidemic_data(disease_type='SARS')
#
# preprocessing_real_epidemic_data(disease_type='COVID')
#
#
#
#
# exit(1)


world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world = world[(world.pop_est > 0) & (world.name != "Antarctica")]
cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))

node_names_all = np.array(
    ['Papua New Guinea', 'Greenland', 'Iceland', 'Canada', 'Algeria', 'Benin', 'Burkina Faso', 'Ghana', "Cote d'Ivoire",
     'Nigeria', 'Niger', 'Tunisia', 'Togo', 'Belgium', 'Germany', 'Estonia', 'Finland', 'United Kingdom', 'Guernsey',
     'Jersey', 'Isle of Man', 'Falkland Islands', 'Netherlands', 'Ireland', 'Denmark', 'Faroe Islands', 'Luxembourg',
     'Norway', 'Poland', 'Sweden', 'South Africa', 'Botswana', 'Congo (Brazzaville)', 'Congo (Kinshasa)', 'Swaziland',
     'Central African Republic', 'Equatorial Guinea', 'Saint Helena', 'Mauritius', 'British Indian Ocean Territory',
     'Cameroon', 'Zambia', 'Comoros', 'Mayotte', 'Reunion', 'Madagascar', 'Angola', 'Gabon', 'Sao Tome and Principe',
     'Mozambique', 'Seychelles', 'Chad', 'Zimbabwe', 'Malawi', 'Lesotho', 'Mali', 'Gambia', 'Spain', 'Sierra Leone',
     'Liberia', 'Morocco', 'Senegal', 'Mauritania', 'Guinea', 'Cape Verde', 'Ethiopia', 'Burundi', 'Somalia', 'Egypt',
     'Kenya', 'Libya', 'Rwanda', 'Sudan', 'South Sudan', 'Tanzania', 'Uganda', 'Albania', 'Bulgaria', 'Cyprus',
     'Croatia', 'France', 'Saint Pierre and Miquelon', 'Greece', 'Hungary', 'Italy', 'Slovenia', 'Czech Republic',
     'Israel', 'Malta', 'Austria', 'Portugal', 'Bosnia and Herzegovina', 'Romania', 'Switzerland', 'Turkey', 'Moldova',
     'Macedonia', 'Gibraltar', 'Serbia', 'Montenegro', 'Slovakia', 'Turks and Caicos Islands', 'Dominican Republic',
     'Guatemala', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Costa Rica', 'El Salvador', 'Haiti', 'Cuba',
     'Cayman Islands', 'Bahamas', 'Belize', 'Cook Islands', 'Fiji', 'Tonga', 'Kiribati', 'Wallis and Futuna', 'Samoa',
     'American Samoa', 'French Polynesia', 'Vanuatu', 'New Caledonia', 'New Zealand', 'Afghanistan', 'Bahrain',
     'Saudi Arabia', 'Iran', 'Jordan', 'Kuwait', 'Lebanon', 'United Arab Emirates', 'Oman', 'Pakistan', 'Iraq', 'Syria',
     'Qatar', 'Northern Mariana Islands', 'Guam', 'Marshall Islands', 'Midway Islands', 'Micronesia', 'Palau', 'Taiwan',
     'Japan', 'South Korea', 'Philippines', 'Argentina', 'Brazil', 'Chile', 'Antarctica', 'Ecuador', 'Paraguay',
     'Colombia', 'Bolivia', 'Suriname', 'French Guiana', 'Peru', 'Uruguay', 'Venezuela', 'Guyana',
     'Antigua and Barbuda', 'Barbados', 'Dominica', 'Martinique', 'Guadeloupe', 'Grenada', 'Virgin Islands',
     'Puerto Rico', 'Saint Kitts and Nevis', 'Saint Lucia', 'Aruba', 'Netherlands Antilles', 'Anguilla',
     'Trinidad and Tobago', 'British Virgin Islands', 'Saint Vincent and the Grenadines', 'Kazakhstan', 'Kyrgyzstan',
     'Azerbaijan', 'Russia', 'Ukraine', 'Belarus', 'Turkmenistan', 'Tajikistan', 'Uzbekistan', 'India', 'Sri Lanka',
     'Cambodia', 'Bangladesh', 'Hong Kong', 'Laos', 'Macau', 'Nepal', 'Bhutan', 'Maldives', 'Thailand', 'Vietnam',
     'Myanmar', 'Indonesia', 'Malaysia', 'Brunei', 'East Timor', 'Singapore', 'Australia', 'Christmas Island',
     'Norfolk Island', 'China', 'North Korea', 'Mongolia', 'United States', 'Latvia', 'Lithuania', 'Armenia', 'Eritrea',
     'Palestine', 'Georgia', 'Yemen', 'Bermuda', 'Solomon Islands', 'Nauru', 'Tuvalu', 'Namibia', 'Djibouti',
     'Montserrat', 'Johnston Atoll', 'Guinea-Bissau', 'Western Sahara', 'Niue', 'Cocos (Keeling) Islands',
     'Wake Island'])

name_idx = np.array([0.000000000000000000e+00,
                     2.000000000000000000e+00,
                     3.000000000000000000e+00,
                     4.000000000000000000e+00,
                     8.000000000000000000e+00,
                     1.100000000000000000e+01,
                     1.300000000000000000e+01,
                     1.400000000000000000e+01,
                     1.500000000000000000e+01,
                     1.600000000000000000e+01,
                     1.700000000000000000e+01,
                     2.200000000000000000e+01,
                     2.300000000000000000e+01,
                     2.400000000000000000e+01,
                     2.600000000000000000e+01,
                     2.700000000000000000e+01,
                     2.800000000000000000e+01,
                     2.900000000000000000e+01,
                     3.000000000000000000e+01,
                     3.800000000000000000e+01,
                     5.700000000000000000e+01,
                     6.000000000000000000e+01,
                     6.400000000000000000e+01,
                     6.500000000000000000e+01,
                     6.800000000000000000e+01,
                     6.900000000000000000e+01,
                     7.000000000000000000e+01,
                     7.500000000000000000e+01,
                     7.700000000000000000e+01,
                     7.800000000000000000e+01,
                     7.900000000000000000e+01,
                     8.000000000000000000e+01,
                     8.200000000000000000e+01,
                     8.300000000000000000e+01,
                     8.400000000000000000e+01,
                     8.500000000000000000e+01,
                     8.600000000000000000e+01,
                     8.700000000000000000e+01,
                     8.800000000000000000e+01,
                     8.900000000000000000e+01,
                     9.000000000000000000e+01,
                     9.100000000000000000e+01,
                     9.200000000000000000e+01,
                     9.300000000000000000e+01,
                     9.400000000000000000e+01,
                     9.600000000000000000e+01,
                     9.800000000000000000e+01,
                     9.900000000000000000e+01,
                     1.000000000000000000e+02,
                     1.020000000000000000e+02,
                     1.030000000000000000e+02,
                     1.040000000000000000e+02,
                     1.050000000000000000e+02,
                     1.060000000000000000e+02,
                     1.070000000000000000e+02,
                     1.080000000000000000e+02,
                     1.090000000000000000e+02,
                     1.100000000000000000e+02,
                     1.120000000000000000e+02,
                     1.130000000000000000e+02,
                     1.140000000000000000e+02,
                     1.160000000000000000e+02,
                     1.170000000000000000e+02,
                     1.210000000000000000e+02,
                     1.230000000000000000e+02,
                     1.240000000000000000e+02,
                     1.250000000000000000e+02,
                     1.260000000000000000e+02,
                     1.280000000000000000e+02,
                     1.290000000000000000e+02,
                     1.300000000000000000e+02,
                     1.310000000000000000e+02,
                     1.320000000000000000e+02,
                     1.330000000000000000e+02,
                     1.340000000000000000e+02,
                     1.350000000000000000e+02,
                     1.370000000000000000e+02,
                     1.380000000000000000e+02,
                     1.390000000000000000e+02,
                     1.450000000000000000e+02,
                     1.470000000000000000e+02,
                     1.480000000000000000e+02,
                     1.490000000000000000e+02,
                     1.500000000000000000e+02,
                     1.510000000000000000e+02,
                     1.520000000000000000e+02,
                     1.540000000000000000e+02,
                     1.550000000000000000e+02,
                     1.560000000000000000e+02,
                     1.570000000000000000e+02,
                     1.580000000000000000e+02,
                     1.600000000000000000e+02,
                     1.610000000000000000e+02,
                     1.620000000000000000e+02,
                     1.630000000000000000e+02,
                     1.640000000000000000e+02,
                     1.650000000000000000e+02,
                     1.660000000000000000e+02,
                     1.670000000000000000e+02,
                     1.700000000000000000e+02,
                     1.710000000000000000e+02,
                     1.730000000000000000e+02,
                     1.740000000000000000e+02,
                     1.750000000000000000e+02,
                     1.770000000000000000e+02,
                     1.780000000000000000e+02,
                     1.830000000000000000e+02,
                     1.840000000000000000e+02,
                     1.890000000000000000e+02,
                     1.900000000000000000e+02,
                     1.910000000000000000e+02,
                     1.920000000000000000e+02,
                     1.930000000000000000e+02,
                     1.940000000000000000e+02,
                     1.960000000000000000e+02,
                     1.990000000000000000e+02,
                     2.000000000000000000e+02,
                     2.010000000000000000e+02,
                     2.020000000000000000e+02,
                     2.030000000000000000e+02,
                     2.040000000000000000e+02,
                     2.060000000000000000e+02,
                     2.070000000000000000e+02,
                     2.100000000000000000e+02,
                     2.130000000000000000e+02,
                     2.140000000000000000e+02,
                     2.150000000000000000e+02,
                     2.180000000000000000e+02,
                     2.200000000000000000e+02,
                     2.210000000000000000e+02,
                     ], dtype=np.int64)

node_names_all = node_names_all[name_idx]
node_names_all = [node_names_all]


def is_node_name(name, name_list):
    flag = False
    for n in name_list:
        if name in n or n in name:
            flag = True
    return flag


def get_index(name, name_list):
    count = -1
    for n in name_list:
        count += 1
        if name in n or n in name:
            break
    print(count)
    return count


for case_no in range(len(node_names_all)):

    ax = world.plot(color='none', edgecolor='gray')
    # ax = world.plot(color='#87CEEB')
    # ax = world.plot(column='pop_est', legend=False)
    ax.set_axis_off()

    node_names = node_names_all[case_no]

    # print(world['name'].tolist())

    count = 0
    made_names = []
    made_points = {}
    for i in range(len(world)):
        if is_node_name(world['name'].iloc[i], node_names):
            made_names.append(world['name'].iloc[i])
            print(world['name'].iloc[i])
            count += 1
            xc, yc = world['geometry'].iloc[i].centroid.x, world['geometry'].iloc[i].centroid.y
            # plt.plot([xc], [yc], marker='o', color="k", alpha=0.7, markersize=10)
            fc_color = (np.random.rand(), np.random.rand(), np.random.rand())
            # fc_color = '#87CEEB'
            if world['geometry'].iloc[i].geometryType() == 'MultiPolygon':
                for geom in world['geometry'].iloc[i].geoms:
                    xs, ys = geom.exterior.xy
                    ax.fill(xs, ys, alpha=0.5, fc=fc_color)
            else:
                geom = world['geometry'].iloc[i]
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, alpha=0.5, fc=fc_color)
            name = world['name'].iloc[i]
            # if name == 'United States of America':
            #     name_ = 'United States'
            # else:
            #     name_ = name

            # ax.text(xc + 3, yc - 10, '%s:' % (node_names.index(name)) + name_, fontsize=8, color='k')
            # ax.text(xc - 3, yc - 3, '%s' % get_index(name, node_names), fontsize=6, color='w')
            #ax.text(xc, yc, '%s' % name, fontsize=6, color='k')
            made_points[get_index(name, node_names)] = (xc, yc)

    for i in range(len(cities)):
        if cities['name'].iloc[i] in node_names:
            made_names.append(cities['name'].iloc[i])
            print(cities['name'].iloc[i])
            count += 1
            xc, yc = cities['geometry'].iloc[i].centroid.x, cities['geometry'].iloc[i].centroid.y
            # plt.plot([xc], [yc], marker='o', color="k", alpha=0.7, markersize=10)
            fc_color = (np.random.rand(), np.random.rand(), np.random.rand())
            # fc_color = '#87CEEB'
            if cities['geometry'].iloc[i].geometryType() == 'MultiPolygon':
                for geom in cities['geometry'].iloc[i].geoms:
                    xs, ys = geom.exterior.xy
                    ax.fill(xs, ys, alpha=0.5, fc=fc_color)
            elif cities['geometry'].iloc[i].geometryType() == 'Polygon':
                geom = cities['geometry'].iloc[i]
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, alpha=0.5, fc=fc_color)
            name = cities['name'].iloc[i]
            # if name == 'United States of America':
            #     name_ = 'United States'
            # else:
            #     name_ = name
            # ax.text(xc + 3, yc - 10, '%s:' % (node_names.index(name)) + name_, fontsize=8, color='k')
            # ax.text(xc - 3, yc - 3, '%s' % get_index(name, node_names), fontsize=6, color='w')
            #ax.text(xc, yc, '%s' % name, fontsize=6, color='k')
            made_points[get_index(name, node_names)] = (xc, yc)

    # print('count = ', count, 'len(node_names)=%s'%len(node_names))
    # print(set(node_names)- set(made_names))

    adj_timeseries_file_names = [
        ('H1N1_adj.txt', 'H1N1_filter_total_cases_raw.txt'),
        ('SARS_adj.txt', 'SARS_filter_total_cases_raw.txt'),
        ('COVID_adj.txt', 'COVID_filter_total_cases_raw.txt'),
    ]

    adj = np.loadtxt(adj_timeseries_file_names[case_no][0])

    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i][j] > 0 and i in list(made_points.keys()) and j in list(made_points.keys()):
                from_point = made_points[j]
                to_point = made_points[i]
                ax.plot([from_point[0], to_point[0]], [from_point[1], to_point[1]], color='gray', alpha=0.5,
                        linewidth=0.2 + adj[i][j] * 15)
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i][j] > 0 and i in list(made_points.keys()) and j in list(made_points.keys()):
                from_point = made_points[j]
                to_point = made_points[i]
                #plt.plot([from_point[0]], [from_point[1]], marker='o', color="k", alpha=0.7, markersize=10)
                ax.text(from_point[0], from_point[1], '%s' % (j), fontsize=6, color='k')

    plt.show()

# exit(1)

# adj_timeseries_file_names = [('Covid19_Spain_adj.txt','Covid19_Spain_total_cases_raw.txt'),
# ('H1N1_adj.txt','H1N1_filter_total_cases_raw.txt'),
# ('SARS_adj.txt','SARS_filter_total_cases_raw.txt'),
# ('COVID_adj.txt','COVID_filter_total_cases_raw.txt'),
#                   ]
adj_timeseries_file_names = [
    ('H1N1_adj.txt', 'H1N1_filter_total_cases_raw.txt'),
    ('SARS_adj.txt', 'SARS_filter_total_cases_raw.txt'),
    ('COVID_adj.txt', 'COVID_filter_total_cases_raw.txt'),
]

sns.set(context='notebook', style='ticks', font_scale=2)

for adj_name, ts_name in adj_timeseries_file_names:

    adj = np.loadtxt(adj_name)
    ts = np.loadtxt(ts_name)

    print(np.sum(adj > 0))

    plt.figure(figsize=(5, 5))
    plt.matshow(adj, cmap=plt.cm.viridis)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('%s.png'%adj_name[:-4])
    plt.show()
    plt.close()

    plt.figure(figsize=(5, 5))
    for i in range(ts.shape[-1]):
        plt.plot(ts[:, i])
    plt.tight_layout()
    plt.savefig('%s.png' % ts_name[:-4])
    plt.show()
    plt.close()
