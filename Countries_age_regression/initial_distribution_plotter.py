import ROOT
import numpy
from array import array


file_name = "smoking_data.csv"

data = numpy.genfromtxt(file_name, delimiter = ',')

#update dataset
for i in range(len(data)):
    data[i][0] = str(i)
    print(data[i][0])



#
arr_gdp, arr_age, arr_cig = array('f'), array('f'), array('f')

#fill default arrays
for i in range(len(data)):
    arr_gdp.append(data[i][2])
    arr_age.append(data[i][3])
    arr_cig.append(data[i][1])
    
#GDP vs AGE
tgr_gdp_age = ROOT.TGraphErrors(len(arr_gdp), arr_gdp, arr_age)
tgr_gdp_age.SetMarkerStyle(20)
tgr_gdp_age.SetMarkerSize(1.)

#GDP vs SIG
tgr_gdp_cig = ROOT.TGraphErrors(len(arr_gdp), arr_gdp, arr_cig)
tgr_gdp_cig.SetMarkerStyle(20)
tgr_gdp_cig.SetMarkerSize(1.)

#AGE vs SIG
tgr_age_cig = ROOT.TGraphErrors(len(arr_age),arr_cig , arr_age)
tgr_age_cig.SetMarkerStyle(20)
tgr_age_cig.SetMarkerSize(1.)

canv = ROOT.TCanvas("c1", "c1", 1200, 400)
canv.Divide(3, 1)
canv.cd(1)
tgr_gdp_age.GetXaxis().SetTitle("GDP, $ per capita")
tgr_gdp_age.GetYaxis().SetTitle("AGE")
tgr_gdp_age.Draw("ap")

canv.cd(2)
tgr_gdp_cig.GetXaxis().SetTitle("GDP, $ per capita")
tgr_gdp_cig.GetYaxis().SetTitle("Sigs per capita per year")
tgr_gdp_cig.Draw("ap")


canv.cd(3)
tgr_age_cig.GetXaxis().SetTitle("Sigs per capita per year")
tgr_age_cig.GetYaxis().SetTitle("AGE")
tgr_age_cig.Draw("ap")
