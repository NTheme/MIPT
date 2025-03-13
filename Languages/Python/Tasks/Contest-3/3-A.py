dev = str(input()).split('_')

row = {"alpha": 1, "beta": 2, "gamma": 3, "delta": 4, "epsilon": 5,
       "zeta": 6, "eta": 7, "theta": 8, "iota": 9, "kappa": 10}
column = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
          "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19,
          "XX": 20, "XXI": 21, "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25, "XXVI": 26, "XXVII": 27, "XXVIII": 28, "XXIX": 29, "XXX": 30}

print(row[dev[0]], column[dev[1]])
