import re

### Function used to translate long tables with station names to a more readable format ###

mapping = {
    "dunkerque_x": "Dunkerque",
    "dunkerque_y": "France full",
    "easington_x": "Birmingham",
    "easington_y": "Easington",
    "germany_x": "Dornum",
    "germany_y": "Emden",
    "germany": "Germany full",
    "nybro": "Nybro",
    "stfergus_x": "Scotland full",
    "stfergus_y": "St Fergus",
    "zeebrugge_x": "Zeebrugge",
    "zeebrugge_y": "Belgium full",
}

top_features_raw = '''
                            Parameter    Mean    Std  Skewness  AC (lag24)  AC (lag48)  Non-Null Count
                                  date     NaN    NaN       NaN         NaN         NaN           39433
        temperature_2m (°C)_stfergus_x    6.54   5.28      0.18        0.89        0.83           39433
   relative_humidity_2m (%)_stfergus_x   87.00  11.64     -1.28        0.66        0.60           39433
          dew_point_2m (°C)_stfergus_x    4.37   4.84     -0.08        0.82        0.72           39433
  apparent_temperature (°C)_stfergus_x    3.48   6.36      0.28        0.89        0.82           39433
                  rain (mm)_stfergus_x    0.14   0.43      7.68        0.09        0.04           39433
         pressure_msl (hPa)_stfergus_x 1011.32  13.49     -0.40        0.79        0.59           39433
            cloud_cover (%)_stfergus_x   83.46  28.71     -1.71        0.18        0.09           39433
      wind_speed_10m (km/h)_stfergus_x   13.01   6.69      0.56        0.36        0.24           39433
     wind_direction_10m (°)_stfergus_x  202.53  91.40     -0.30        0.32        0.21           39433
                  is_day ()_stfergus_x    0.52   0.50     -0.07        0.99        0.99           39433
 shortwave_radiation (W/m²)_stfergus_x  100.48 166.90      1.95        0.82        0.81           39433
    direct_radiation (W/m²)_stfergus_x   47.38 112.79      3.12        0.58        0.53           39433
        temperature_2m (°C)_stfergus_y    9.01   4.73      0.07        0.86        0.80           39433
   relative_humidity_2m (%)_stfergus_y   81.92  11.00     -0.66        0.47        0.36           39433
          dew_point_2m (°C)_stfergus_y    5.95   4.78     -0.09        0.79        0.71           39433
  apparent_temperature (°C)_stfergus_y    5.77   5.99      0.20        0.87        0.81           39433
                  rain (mm)_stfergus_y    0.11   0.37      9.17        0.03        0.04           39433
         pressure_msl (hPa)_stfergus_y 1010.70  13.52     -0.42        0.79        0.58           39433
            cloud_cover (%)_stfergus_y   72.19  35.28     -0.91        0.12        0.07           39433
      wind_speed_10m (km/h)_stfergus_y   16.36   8.12      0.76        0.29        0.16           39433
     wind_direction_10m (°)_stfergus_y  208.50  86.44     -0.30        0.27        0.16           39433
                  is_day ()_stfergus_y    0.52   0.50     -0.07        0.99        0.99           39433
 shortwave_radiation (W/m²)_stfergus_y  119.34 185.75      1.64        0.88        0.86           39433
    direct_radiation (W/m²)_stfergus_y   63.79 127.30      2.44        0.66        0.62           39433
             temperature_2m (°C)_nybro    9.01   6.42      0.06        0.92        0.88           39433
        relative_humidity_2m (%)_nybro   81.24  14.15     -1.01        0.68        0.59           39433
               dew_point_2m (°C)_nybro    5.68   5.84     -0.25        0.85        0.78           39433
       apparent_temperature (°C)_nybro    5.77   7.72      0.21        0.93        0.88           39433
                       rain (mm)_nybro    0.11   0.39      7.45        0.05        0.04           39433
              pressure_msl (hPa)_nybro 1012.82  11.47     -0.40        0.78        0.56           39433
                 cloud_cover (%)_nybro   69.51  37.70     -0.78        0.23        0.14           39433
           wind_speed_10m (km/h)_nybro   16.66   7.94      0.65        0.37        0.19           39433
          wind_direction_10m (°)_nybro  200.57  87.58     -0.42        0.40        0.25           39433
                       is_day ()_nybro    0.52   0.50     -0.08        0.99        0.99           39433
      shortwave_radiation (W/m²)_nybro  119.51 189.52      1.71        0.87        0.85           39433
         direct_radiation (W/m²)_nybro   68.33 138.50      2.43        0.70        0.65           39433
       temperature_2m (°C)_zeebrugge_x   11.51   5.82      0.12        0.91        0.85           39433
  relative_humidity_2m (%)_zeebrugge_x   78.64  12.51     -0.87        0.56        0.46           39433
         dew_point_2m (°C)_zeebrugge_x    7.70   5.30     -0.30        0.84        0.75           39433
 apparent_temperature (°C)_zeebrugge_x    8.49   7.32      0.27        0.91        0.85           39433
                 rain (mm)_zeebrugge_x    0.11   0.39      8.20        0.04        0.04           39433
        pressure_msl (hPa)_zeebrugge_x 1015.45  10.67     -0.50        0.78        0.55           39433
           cloud_cover (%)_zeebrugge_x   65.53  38.71     -0.59        0.22        0.14           39433
     wind_speed_10m (km/h)_zeebrugge_x   18.51   9.33      0.73        0.43        0.28           39433
    wind_direction_10m (°)_zeebrugge_x  191.08  95.36     -0.35        0.35        0.21           39433
                 is_day ()_zeebrugge_x    0.51   0.50     -0.06        1.00        0.99           39433
shortwave_radiation (W/m²)_zeebrugge_x  139.00 214.15      1.59        0.89        0.87           39433
   direct_radiation (W/m²)_zeebrugge_x   85.35 162.06      2.15        0.74        0.71           39433
       temperature_2m (°C)_zeebrugge_y   10.89   6.76      0.17        0.91        0.85           39433
  relative_humidity_2m (%)_zeebrugge_y   78.18  15.91     -0.88        0.72        0.65           39433
         dew_point_2m (°C)_zeebrugge_y    6.81   5.51     -0.35        0.84        0.75           39433
 apparent_temperature (°C)_zeebrugge_y    8.08   8.13      0.26        0.92        0.86           39433
                 rain (mm)_zeebrugge_y    0.10   0.38      8.82        0.05        0.04           39433
        pressure_msl (hPa)_zeebrugge_y 1016.14  10.11     -0.48        0.77        0.53           39433
           cloud_cover (%)_zeebrugge_y   67.35  38.96     -0.70        0.28        0.17           39433
     wind_speed_10m (km/h)_zeebrugge_y   15.69   8.04      0.85        0.47        0.33           39433
    wind_direction_10m (°)_zeebrugge_y  191.03  89.71     -0.45        0.34        0.18           39433
                 is_day ()_zeebrugge_y    0.51   0.50     -0.04        1.00        0.99           39433
shortwave_radiation (W/m²)_zeebrugge_y  135.11 208.31      1.61        0.88        0.86           39433
   direct_radiation (W/m²)_zeebrugge_y   79.83 154.89      2.25        0.73        0.68           39433
         temperature_2m (°C)_germany_x   10.54   5.87      0.14        0.92        0.87           39433
    relative_humidity_2m (%)_germany_x   80.34  11.58     -0.71        0.49        0.38           39433
           dew_point_2m (°C)_germany_x    7.12   5.58     -0.18        0.86        0.78           39433
   apparent_temperature (°C)_germany_x    7.27   7.51      0.27        0.92        0.87           39433
                   rain (mm)_germany_x    0.11   0.38      8.49        0.04        0.03           39433
          pressure_msl (hPa)_germany_x 1014.33  10.92     -0.39        0.77        0.55           39433
             cloud_cover (%)_germany_x   68.35  37.54     -0.72        0.21        0.14           39433
       wind_speed_10m (km/h)_germany_x   19.24   9.40      0.75        0.42        0.26           39433
      wind_direction_10m (°)_germany_x  197.24  94.05     -0.33        0.33        0.19           39433
                   is_day ()_germany_x    0.51   0.50     -0.06        1.00        0.99           39433
  shortwave_radiation (W/m²)_germany_x  133.00 206.34      1.60        0.89        0.88           39433
     direct_radiation (W/m²)_germany_x   80.25 153.47      2.19        0.74        0.72           39433
         temperature_2m (°C)_germany_y   10.66   6.16      0.17        0.92        0.87           39433
    relative_humidity_2m (%)_germany_y   80.10  12.79     -0.83        0.60        0.51           39433
           dew_point_2m (°C)_germany_y    7.13   5.56     -0.19        0.86        0.78           39433
   apparent_temperature (°C)_germany_y    7.68   7.69      0.29        0.92        0.87           39433
                   rain (mm)_germany_y    0.11   0.37      7.89        0.03        0.04           39433
          pressure_msl (hPa)_germany_y 1014.50  10.85     -0.39        0.77        0.55           39433
             cloud_cover (%)_germany_y   69.03  37.36     -0.76        0.21        0.15           39433
       wind_speed_10m (km/h)_germany_y   16.97   8.41      0.74        0.43        0.28           39433
      wind_direction_10m (°)_germany_y  196.52  93.21     -0.33        0.33        0.19           39433
                   is_day ()_germany_y    0.51   0.50     -0.05        1.00        0.99           39433
  shortwave_radiation (W/m²)_germany_y  129.28 200.55      1.63        0.88        0.87           39433
     direct_radiation (W/m²)_germany_y   75.00 146.36      2.29        0.72        0.69           39433
           temperature_2m (°C)_germany    8.60   7.54      0.14        0.93        0.87           39433
      relative_humidity_2m (%)_germany   77.52  16.66     -0.85        0.73        0.67           39433
             dew_point_2m (°C)_germany    4.42   6.10     -0.23        0.86        0.77           39433
     apparent_temperature (°C)_germany    5.83   8.85      0.20        0.93        0.87           39433
                     rain (mm)_germany    0.08   0.32     11.05        0.05        0.02           39433
            pressure_msl (hPa)_germany 1016.47   9.59     -0.27        0.77        0.52           39433
               cloud_cover (%)_germany   69.90  37.66     -0.83        0.29        0.21           39433
         wind_speed_10m (km/h)_germany   11.79   6.11      0.82        0.46        0.32           39433
        wind_direction_10m (°)_germany  201.91  87.98     -0.39        0.31        0.18           39433
                     is_day ()_germany    0.51   0.50     -0.04        1.00        0.99           39433
    shortwave_radiation (W/m²)_germany  133.75 207.06      1.60        0.88        0.86           39433
       direct_radiation (W/m²)_germany   77.66 152.74      2.26        0.72        0.66           39433
       temperature_2m (°C)_dunkerque_x   11.64   5.63      0.11        0.90        0.84           39433
  relative_humidity_2m (%)_dunkerque_x   78.90  12.21     -0.81        0.58        0.48           39433
         dew_point_2m (°C)_dunkerque_x    7.88   5.13     -0.35        0.83        0.74           39433
 apparent_temperature (°C)_dunkerque_x    8.71   7.03      0.25        0.91        0.85           39433
                 rain (mm)_dunkerque_x    0.10   0.38      9.50        0.03        0.04           39433
        pressure_msl (hPa)_dunkerque_x 1015.56  10.72     -0.54        0.78        0.55           39433
           cloud_cover (%)_dunkerque_x   64.70  38.89     -0.56        0.19        0.12           39433
     wind_speed_10m (km/h)_dunkerque_x   17.88   9.10      0.75        0.42        0.28           39433
    wind_direction_10m (°)_dunkerque_x  186.16  92.49     -0.34        0.39        0.24           39433
                 is_day ()_dunkerque_x    0.52   0.50     -0.06        1.00        0.99           39433
shortwave_radiation (W/m²)_dunkerque_x  140.79 215.04      1.56        0.88        0.87           39433
   direct_radiation (W/m²)_dunkerque_x   86.99 162.62      2.10        0.73        0.70           39433
       temperature_2m (°C)_dunkerque_y   12.62   7.32      0.23        0.92        0.86           39433
  relative_humidity_2m (%)_dunkerque_y   74.91  16.93     -0.74        0.72        0.65           39433
         dew_point_2m (°C)_dunkerque_y    7.75   5.53     -0.29        0.84        0.74           39433
 apparent_temperature (°C)_dunkerque_y   10.73   8.73      0.25        0.93        0.86           39433
                 rain (mm)_dunkerque_y    0.10   0.41     10.61        0.05        0.03           39433
        pressure_msl (hPa)_dunkerque_y 1017.67   8.53     -0.47        0.78        0.54           39433
           cloud_cover (%)_dunkerque_y   62.89  40.37     -0.50        0.31        0.19           39433
     wind_speed_10m (km/h)_dunkerque_y   11.12   5.58      0.98        0.37        0.22           39433
    wind_direction_10m (°)_dunkerque_y  185.10  96.49     -0.24        0.29        0.18           39433
                 is_day ()_dunkerque_y    0.51   0.50     -0.05        1.00        0.99           39433
shortwave_radiation (W/m²)_dunkerque_y  151.18 227.65      1.52        0.89        0.87           39433
   direct_radiation (W/m²)_dunkerque_y   93.75 174.59      2.08        0.75        0.70           39433
       temperature_2m (°C)_easington_x   10.29   5.91      0.20        0.89        0.83           39433
  relative_humidity_2m (%)_easington_x   80.23  14.05     -0.90        0.71        0.67           39433
         dew_point_2m (°C)_easington_x    6.73   4.85     -0.19        0.78        0.69           39433
 apparent_temperature (°C)_easington_x    7.21   7.04      0.32        0.90        0.83           39433
                 rain (mm)_easington_x    0.09   0.37     12.39        0.02        0.02           39433
        pressure_msl (hPa)_easington_x 1014.28  12.01     -0.51        0.79        0.57           39433
           cloud_cover (%)_easington_x   69.57  37.47     -0.79        0.15        0.09           39433
     wind_speed_10m (km/h)_easington_x   16.67   8.22      0.68        0.42        0.25           39433
    wind_direction_10m (°)_easington_x  197.72  91.57     -0.50        0.41        0.26           39433
                 is_day ()_easington_x    0.51   0.50     -0.06        1.00        0.99           39433
shortwave_radiation (W/m²)_easington_x  124.76 191.97      1.66        0.86        0.84           39433
   direct_radiation (W/m²)_easington_x   68.22 136.56      2.49        0.66        0.60           39433
       temperature_2m (°C)_easington_y   10.45   5.18      0.10        0.88        0.82           39433
  relative_humidity_2m (%)_easington_y   80.78  11.13     -0.78        0.57        0.47           39433
         dew_point_2m (°C)_easington_y    7.12   4.81     -0.14        0.81        0.73           39433
 apparent_temperature (°C)_easington_y    7.10   6.45      0.21        0.89        0.83           39433
                 rain (mm)_easington_y    0.08   0.33      8.36        0.03        0.01           39433
        pressure_msl (hPa)_easington_y 1013.50  12.33     -0.48        0.79        0.57           39433
           cloud_cover (%)_easington_y   67.75  37.96     -0.71        0.14        0.08           39433
     wind_speed_10m (km/h)_easington_y   19.17   9.29      0.62        0.38        0.23           39433
    wind_direction_10m (°)_easington_y  194.39  89.95     -0.43        0.39        0.27           39433
                 is_day ()_easington_y    0.51   0.50     -0.06        1.00        0.99           39433
shortwave_radiation (W/m²)_easington_y  133.59 202.92      1.57        0.89        0.87           39433
   direct_radiation (W/m²)_easington_y   75.57 145.90      2.29        0.69        0.64           39433
                                    DK   53.48  50.75      1.85        0.99        0.99           39433
                                    DE   52.63  50.72      1.88        0.99        0.99           39433
                                    FR   46.30  39.13      1.46        0.99        0.98           39433
                                    BE   47.85  41.85      1.62        0.99        0.97           39433
                                    UK   42.53  35.14      1.72        0.98        0.97           39433
                                  Mean   48.56  42.86      1.66        0.99        0.98           39433
                                DK_rel    1.07   0.09      2.42        0.94        0.90           39433
                                DE_rel    1.04   0.09      2.88        0.96        0.91           39433
                                FR_rel    0.97   0.05     -2.86        0.93        0.85           39433
                                BE_rel    0.99   0.05     -2.32        0.86        0.75           39433
                                UK_rel    0.93   0.14     -2.26        0.94        0.91           39433
                            DK_rel_pct    7.03   9.25      2.42        0.94        0.90           39433
                            DE_rel_pct    4.22   8.99      2.88        0.96        0.91           39433
                            FR_rel_pct   -3.15   5.44     -2.86        0.93        0.85           39433
                            BE_rel_pct   -1.44   5.41     -2.32        0.86        0.75           39433
                            UK_rel_pct   -6.66  13.64     -2.26        0.94        0.91           39433
                            DK_is_best    0.65   0.47     -0.64        0.72        0.59           39433
                            DE_is_best    0.13   0.33      2.19        0.64        0.45           39433
                            FR_is_best    0.01   0.08     11.50        0.35        0.11           39433
                            BE_is_best    0.01   0.10      9.65        0.30        0.14           39433
                            UK_is_best    0.20   0.39      1.50        0.76        0.67           39433
            Volumrate_Easington_hourly   51.55  22.55     -0.73        0.91        0.87           39433
            Volumrate_St_Fergus_hourly    5.25   7.08      2.38        0.83        0.76           39433
              Volumrate_Danmark_hourly    9.19  10.22      0.73        0.98        0.98           39433
               Volumrate_Belgia_hourly   42.01   3.94     -5.18        0.74        0.61           39433
            Volumrate_Frankrike_hourly   44.60  10.13     -2.04        0.83        0.73           39433
             Volumrate_Tyskland_hourly  138.15  15.37     -1.86        0.85        0.79           39433
                                  year   21.78   1.31      0.12        1.00        1.00           39433
                              hour_sin    0.00   0.71     -0.00        1.00        1.00           39433
                              hour_cos   -0.00   0.71      0.00        1.00        1.00           39433
                               dow_sin   -0.00   0.71      0.00        0.62       -0.22           39433
                               dow_cos   -0.00   0.71      0.00        0.62       -0.22           39433
                             month_sin    0.06   0.70     -0.15        1.00        0.99           39433
                             month_cos   -0.02   0.71      0.04        1.00        0.99           39433
                            is_holiday    0.04   0.18      5.05        0.33        0.07           39433
         Nominasjoner Easington [MSM3]   51.21  22.82     -0.78        0.94        0.90           39433
        Nominasjoner St. Fergus [MSM3]    4.86   7.12      2.30        0.88        0.80           39433
           Nominasjoner Danmark [MSM3]    8.06  10.96      0.72        0.99        0.99           39433
            Nominasjoner Belgia [MSM3]   41.89   4.07     -5.70        0.78        0.62           39433
         Nominasjoner Frankrike [MSM3]   44.48  10.15     -2.20        0.85        0.75           39433
          Nominasjoner Tyskland [MSM3]  138.35  15.21     -1.88        0.87        0.82           39433'''.strip().split('\n')  # Replace ... with all remaining entries

import re

# Mapping from suffix to location
mapping = {
    "dunkerque_x": "Dunkerque",
    "dunkerque_y": "France full",
    "easington_x": "Birmingham",
    "easington_y": "Easington",
    "germany_x": "Dornum",
    "germany_y": "Emden",
    "germany": "Germany full",
    "nybro": "Nybro",
    "stfergus_x": "Scotland full",
    "stfergus_y": "St Fergus",
    "zeebrugge_x": "Zeebrugge",
    "zeebrugge_y": "Belgium full",
}

lines = top_features_raw

translated = []

for line in lines:
    match = re.match(r'\s*(.+?)\s+[-\d\.]+', line)
    if not match:
        continue
    feature = match.group(1).strip()

    translated_feature = feature
    for suffix, location in mapping.items():
        if feature.endswith(f"_{suffix}"):
            translated_feature = feature.replace(f"_{suffix}", f" ({location})")
            break
        elif feature == suffix:
            translated_feature = location
            break

    translated.append(translated_feature)

print("Translated Top Features:\n")
for i, feat in enumerate(translated, 1):
    print(f"{i}. {feat}")
