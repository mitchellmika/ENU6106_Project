import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import time

matplotlib.rcParams.update({'font.size': 20})

def main(inputFile):
    # Read input file
    with open(inputFile, mode="r") as file:
        readlines = file.readlines()
        inpDict = {}

        for line in readlines:
            if "#" not in line and line[0] != "\n":
                line = line.replace("\n", "")
                line = line.split(" ")
                if len(line) == 2:
                    inpDict[line[0]] = line[1]
                else:
                    inpDict[line[0]] = line[1:]
    
    # Process input dictionary
    power = float(inpDict["Power"])
    P = float(inpDict["Pitch"])
    D_fuel = float(inpDict["D_Fuel"])
    L_assem = float(inpDict["L_Assem"])

    fuelMeshesPerPin = int(inpDict["Fuel_Meshes"])
    waterMeshesPerPin = int(inpDict["Water_Meshes"])

    geometry = []
    geometry.extend(inpDict["BC_Left"])
    assemGeoms = inpDict["Assem1"] + inpDict["Assem2"]
    for i in range(0, len(assemGeoms), 2):
        geometry.extend(assemGeoms[i] * int(assemGeoms[i+1]))
    geometry.extend(inpDict["BC_Right"])

    dataSet = inpDict["XS_Set"]

    runMC = bool(int(inpDict["Run_MC"]))
    numHistories = int(inpDict["Histories"])
    numGenerations = int(inpDict["Generations"])
    skipGenerations = int(inpDict["Skipped_Gens"])

    runFD = bool(int(inpDict["Run_FD"]))

    # Read data file
    with open(f"xs_set{dataSet}.csv", mode="r") as file:
        csvFile = csv.reader(file)
        xsDict = {}
        for line in csvFile:
            xsDict[line[0]] = {
                "Sigma_tr1":float(line[1]),
                "D_1": 1 / (3*float(line[1])),
                "Sigma_s11":float(line[2]),
                "Sigma_s12":float(line[3]),
                "Sigma_a1":float(line[4]),
                "Sigma_r1":float(line[4]) + float(line[3]),
                "Sigma_f1":float(line[5]),
                "Sigma_c1":float(line[4]) - float(line[5]),
                "Sigma_t1":float(line[2]) + float(line[3]) + float(line[4]),
                "Capture_1":(float(line[4]) - float(line[5])) / (float(line[2]) + float(line[3]) + float(line[4])), # Upper cdf bound for capture interaction in g1
                "Fission_1":float(line[4]) / (float(line[2]) + float(line[3]) + float(line[4])), # Upper cdf bound for fission in g1
                "InScatter_1":(float(line[4]) + float(line[2])) / (float(line[2]) + float(line[3]) + float(line[4])), # Upper cdf bound for inscatter in g1
                "Nu_f1":float(line[6]),
                "Chi_1":float(line[7]),
                "Sigma_tr2":float(line[8]),
                "D_2": 1 / (3*float(line[8])),
                "Sigma_s22":float(line[9]),
                "Sigma_s21":float(line[10]),
                "Sigma_a2":float(line[11]),
                "Sigma_r2":float(line[10]) + float(line[11]),
                "Sigma_f2":float(line[12]),
                "Sigma_c2":float(line[11]) - float(line[12]),
                "Sigma_t2":float(line[9]) + float(line[10]) + float(line[11]),
                "Capture_2": (float(line[11]) - float(line[12])) / (float(line[9]) + float(line[10]) + float(line[11])),
                "Fission_2":float(line[11]) / ((float(line[9]) + float(line[10]) + float(line[11]))),
                "InScatter_2":(float(line[11]) + float(line[9]))  / ((float(line[9]) + float(line[10]) + float(line[11]))),
                "Nu_f2":float(line[13]),
                "Chi_2":float(line[14])
            }
    
    # Define geometry
    if waterMeshesPerPin % 2 != 0:
        raise RuntimeError("Num water meshes must be divisible by 2 in order to accomodate half pin")
    D_water = P - D_fuel
    meshSize = {"W": (D_water / waterMeshesPerPin), "M":(D_fuel / fuelMeshesPerPin), "U":(D_fuel / fuelMeshesPerPin), "C":(D_fuel / fuelMeshesPerPin)}

    geom = []
    if geometry[0] == "W":
        mfp = 1.0 / xsDict["W"]["Sigma_t1"]
        refSize = 5 * mfp
        refCells = int(refSize / meshSize["W"])
        geom.extend("W" * refCells)

    for pin in geometry[1:-1]:
        geom.extend("W" * int(waterMeshesPerPin/2) )
        geom.extend(pin * fuelMeshesPerPin)
        geom.extend("W" * int(waterMeshesPerPin/2) )
    
    if geometry[-1] == "W":
        mfp = 1.0 / xsDict["W"]["Sigma_t1"]
        refSize = 5 * mfp
        refCells = int(refSize / meshSize["W"])
        geom.extend("W" * refCells)

    rightBounds = [meshSize[geom[0]]]
    leftBounds = [0]
    for i in range(len(geom)-1):
        leftBounds.append(rightBounds[i])
        rightBounds.append(rightBounds[i] + meshSize[geom[i+1]])
    
    L_geom = rightBounds[-1]

    # Functions for finding left and right bounds given cell index
    cellRightBound = lambda index: rightBounds[index]
    cellLeftBound = lambda index: leftBounds[index]
    cellCenter = lambda index: (leftBounds[index] + rightBounds[index]) / 2.0

    # Function for find index based on position
    def posToIndex(pos):
        if pos < cellLeftBound(0):
            return -1
        elif pos > cellRightBound(-1):
            return len(geom)
        for ind in range(len(geom)):
            if cellLeftBound(ind) < pos < cellRightBound(ind):
                return ind
    
    if runMC:
        # Define fission distribution
        fuelCells = np.where(np.array(geom) != "W")[0]
        UCells = np.where(np.array(geom) == "U")[0]
        MCells = np.where(np.array(geom) == "M")[0]
        numUCells = len(UCells)
        numMCells = len(MCells)

        normalizingFactor = (numUCells * ( (xsDict["U"]["Sigma_f1"] * xsDict["U"]["Nu_f1"]) +  (xsDict["U"]["Sigma_f2"] * xsDict["U"]["Nu_f2"]))) + (numMCells * ((xsDict["M"]["Sigma_f1"] * xsDict["M"]["Nu_f1"]) + (xsDict["M"]["Sigma_f2"] * xsDict["M"]["Nu_f2"])))
        F_U =  ( (xsDict["U"]["Sigma_f1"] * xsDict["U"]["Nu_f1"]) +  (xsDict["U"]["Sigma_f2"] * xsDict["U"]["Nu_f2"])) / normalizingFactor
        F_M = ((xsDict["M"]["Sigma_f1"] * xsDict["M"]["Nu_f1"]) + (xsDict["M"]["Sigma_f2"] * xsDict["M"]["Nu_f2"])) / normalizingFactor
        
        xsDict["M"]["F_i"] = F_M
        xsDict["U"]["F_i"] = F_U
        xsDict["W"]["F_i"] = 0

        F = np.empty(len(geom))
        for i in range(len(geom)):
            if geom[i] == "W":
                F[i] = 0
            elif geom[i] == "U":
                F[i] = F_U
            elif geom[i] == "M":
                F[i] = F_M
        
        if abs(np.sum(F) - 1.0) > 0.0001:
            raise RuntimeError("Fission start source should never be less than one")

        # Zeros arrays for track length tallying
        trackLengths = {1:np.zeros( (numGenerations,len(geom) )), 2:np.zeros((numGenerations,len(geom)))}
        currents = {1:np.zeros((numGenerations,len(geom)+1)), 2:np.zeros((numGenerations,len(geom)+1))}
        centerCurrents = {1:np.zeros((numGenerations,len(geom))), 2:np.zeros((numGenerations,len(geom)))}

        flux1 = np.zeros( (numGenerations,len(geom) ))
        flux2 = np.zeros( (numGenerations,len(geom) ))

        # Begin generation loop
        starts = []
        k = np.ones(numGenerations+1)

        startTime = time.time()
        for i in range(numGenerations):
            print(f"Generation {i}: k={k[i]:.4f}")
            # Begin history loop
            for j in range(numHistories):
                # Determine cell start index
                cellIndex = None
                startCellRandNum = np.random.random(1)[0]
                for cell in fuelCells:
                    startCellRandNum -= F[cell]
                    if startCellRandNum < 0:
                        cellIndex = cell
                        
                        break
                
                # Determine start position
                pos = np.random.uniform(cellLeftBound(cellIndex), cellRightBound(cellIndex))
                starts.append(pos)
                energyGroup = 1

                absorbed = False
                resampleDir = True
                resampleDist = True
                while not absorbed:
                    material = geom[cellIndex]

                    if resampleDir:
                        randDir = np.random.random(1)[0]
                        mu = 2*randDir - 1

                    if resampleDist:
                        squiggly = np.random.random(1)[0]
                        travelDistance = (-math.log(squiggly) / xsDict[material][f"Sigma_t{energyGroup}"]) * mu

                    # Update position and cell index (assuming no mat or problem boundary crossed, will check later)
                    newPos = travelDistance + pos
                    newCellIndex = posToIndex(newPos)

                    matChange = False
                    resampleDir = True
                    resampleDist = True

                    ##########################################################################################################################################################
                    # Check for material crosing along the way
                    # If material changes, need to add track lengths and resample travel distance at material boundary
                    #########################################################################################################################################################
                    # For right to left
                    if travelDistance < 0:
                        edgeCell = 0 if newCellIndex < 0 else newCellIndex-1 # Use to handle cases where particle crosses problem boundary

                        rng = range(cellIndex,edgeCell, -1)
                        for cell in rng: # iterate from travelled distance from right to left before original cell to find where material change occurs
                            if geom[cell] is not geom[cellIndex]: # Enter if material change found along path
                                newCellIndex = cell

                                # Update track lengths
                                trackLengths[energyGroup][i][newCellIndex+1:cellIndex] += (meshSize[material]/abs(mu)) # Add track length to all cells until material change
                                trackLengths[energyGroup][i][cellIndex] += ((pos - cellLeftBound(cellIndex))/abs(mu)) # Add track length to original cell
                                pos = cellRightBound(newCellIndex) # Move neutron to where material boundary was crossed  (right edge of cell bc moving right to left)

                                # Update currents
                                firstEdge = cellIndex
                                lastEdge = newCellIndex +1
                                currents[energyGroup][i][lastEdge:firstEdge+1] += mu

                                cellIndex = newCellIndex
                                resampleDir = False
                                resampleDist = True
                                matChange=True
                                break

                    # For left to right
                    else:
                        edgeCell = len(geom)-1 if (newCellIndex > len(geom)-1) else newCellIndex # Use to handle cases where particle crosses problem boundary

                        rng = range(cellIndex,edgeCell+1)
                        for cell in rng: # iterate from travelled distance from right to left before original cell to find where material change occurs
                            if geom[cell] is not geom[cellIndex]: # Enter if material change found along path
                                newCellIndex = cell

                                # Update track lengths
                                trackLengths[energyGroup][i][cellIndex+1:newCellIndex] += (meshSize[material]/abs(mu)) # Add track length to all cells until material change
                                trackLengths[energyGroup][i][cellIndex] += ((cellRightBound(cellIndex) - pos)/abs(mu)) # Add track length to original cell
                                pos = cellLeftBound(newCellIndex) # Move neutron to boundary (left bound bc traveling left to right)

                                # Update currents
                                firstEdge = cellIndex + 1
                                lastEdge = newCellIndex
                                currents[energyGroup][i][firstEdge:lastEdge+1] += mu

                                cellIndex = newCellIndex
                                resampleDir = False
                                resampleDist = True
                                matChange=True
                                break

                    if matChange:
                            continue  
                    #######################################################################################################################################################
                    # Check for problem boundary crossing
                    #######################################################################################################################################################
                    if newPos < 0 and geometry[0] == "V":
                        # Particle is lost to vacuum on left side
                        trackLengths[energyGroup][i][cellIndex] += ((pos - cellLeftBound(cellIndex))/abs(mu))
                        if cellIndex != 0: 
                            trackLengths[energyGroup][i][0:cellIndex] += (meshSize[material]/abs(mu))

                        # Update currents
                        firstEdge = cellIndex
                        lastEdge = 0
                        currents[energyGroup][i][lastEdge:firstEdge+1] += mu
                        absorbed = True
                        resampleDir = True
                        resampleDist = True
                        continue

                    elif newPos < 0 and geometry[0] == "I":
                        # Particle is reflected on left side

                        # Update tracklengths for travel to boundary
                        trackLengths[energyGroup][i][cellIndex] += ((pos - cellLeftBound(cellIndex))/abs(mu))
                        if cellIndex != 0: 
                            trackLengths[energyGroup][i][0:cellIndex] += ((meshSize[material])/abs(mu))

                        # Update currents for travel to boundary
                        firstEdge = cellIndex
                        lastEdge = 1
                        currents[energyGroup][i][lastEdge:firstEdge+1] += mu

                        # Calculate reflected travel distance, restart history loop
                        travelDistance += pos
                        pos = 0.0
                        cellIndex = 0
                        travelDistance *= -1
                        mu *= -1
                        resampleDir = False
                        resampleDist = False
                        continue

                    elif newPos > L_geom and geometry[-1] == "V":
                        # Particle is lost to vacuum on right side
                        trackLengths[energyGroup][i][cellIndex] += ((cellRightBound(cellIndex) - pos)/abs(mu)) # Add track length to start cell
                        if cellIndex != len(geom)-1: 
                            trackLengths[energyGroup][i][cellIndex+1:len(geom)] += (meshSize[material]/abs(mu)) # Add track length to other cells if start cell was not at boundary
                        
                        # Update currents
                        firstEdge = cellIndex + 1
                        lastEdge = len(geom)
                        currents[energyGroup][i][firstEdge:lastEdge+1] += mu
                        absorbed = True
                        resampleDir = True
                        resampleDist = True
                        continue

                    elif newPos > (L_geom) and geometry[-1] == "I":
                        # Particle is reflected on right side
                        trackLengths[energyGroup][i][cellIndex] += ((cellRightBound(cellIndex) - pos) / abs(mu))
                        if cellIndex != len(geom)-1: 
                            trackLengths[energyGroup][i][cellIndex+1:len(geom)] += ((meshSize[material])/abs(mu)) # Add track length to other cells if start cell was not at boundary

                        # Update currents
                        firstEdge = cellIndex + 1
                        lastEdge = len(geom)-1
                        currents[energyGroup][i][firstEdge:lastEdge+1] += mu

                        # Calculate reflected travel distance, restart history loop
                        travelDistance -= (L_geom - pos)
                        pos = L_geom
                        cellIndex = len(geom) - 1
                        travelDistance *= -1
                        mu *= -1
                        resampleDir = False
                        resampleDist = False
                        continue
                    
                    ##################################################################################################################################################
                    # No material changes, no problem boundaries crossed
                    # Compute tally lengths and sample interaction
                    #################################################################################################################################################

                    # Right to left
                    if newCellIndex != cellIndex and travelDistance < 0:
                        # Update track lengths
                        trackLengths[energyGroup][i][cellIndex] += ((pos - cellLeftBound(cellIndex))/abs(mu)) # Add track length to original cell
                        trackLengths[energyGroup][i][newCellIndex+1:cellIndex] += (meshSize[material]/abs(mu)) # Add track length to cells along the way
                        trackLengths[energyGroup][i][newCellIndex] += ((cellRightBound(newCellIndex) - newPos)/abs(mu)) # Add track length to new cell location

                        # Update currents
                        firstEdge = cellIndex
                        lastEdge = newCellIndex + 1
                        currents[energyGroup][i][lastEdge:firstEdge+1] += mu

                    # Left to right
                    elif newCellIndex != cellIndex and travelDistance > 0:
                        # Update track lengths
                        trackLengths[energyGroup][i][cellIndex] += ((cellRightBound(cellIndex) - pos)/abs(mu)) # Add track length to original cell
                        trackLengths[energyGroup][i][cellIndex+1:newCellIndex] += (meshSize[material]/abs(mu)) # Add track length to cells along the way
                        trackLengths[energyGroup][i][newCellIndex] += ((newPos - cellLeftBound(newCellIndex))/abs(mu)) # Add track length to new cell location

                        # Update currents
                        firstEdge = cellIndex + 1
                        lastEdge = newCellIndex
                        currents[energyGroup][i][firstEdge:lastEdge+1] += mu
                    else:
                        trackLengths[energyGroup][i][cellIndex] += (abs(travelDistance))

                    pos = newPos
                    cellIndex = newCellIndex
                    if cellIndex < 0 or cellIndex >= len(geom): raise RuntimeError("Cell error")

                    interactionSample = np.random.random(1)[0]
                    if interactionSample < xsDict[material][f"Capture_{energyGroup}"]:
                        # Capture interaction, end history
                        absorbed = True
                    elif xsDict[material][f"Capture_{energyGroup}"] <= interactionSample < xsDict[material][f"Fission_{energyGroup}"]:
                        # Fission interaction, end history
                        absorbed = True
                    elif xsDict[material][f"Fission_{energyGroup}"] <= interactionSample < xsDict[material][f"InScatter_{energyGroup}"]:
                        # In-scattering, no change
                        pass
                    else:
                        # Down-scattering
                        # Check if up-scattering has occured (not allowed, throw error)
                        if energyGroup == 2:
                            raise RuntimeError("Up-scattering has occured, check calculated cdf")
                        else:
                            energyGroup = 2
                    
                    resampleDir = True
                    resampleDist = True
                    
            ##########################################################################################################
            # History loop ends
            ##########################################################################################################

            # Calculate fluxes for current generation
            for j in range(len(geom)):
                flux1[i][j] = trackLengths[1][i][j] / (k[i] * meshSize[geom[j]] * numHistories)
                flux2[i][j] = trackLengths[2][i][j] / (k[i] * meshSize[geom[j]] * numHistories)

            # Calculate center currents for current generation
            for j in range(len(geom)):
                centerCurrents[1][i][j] = (((currents[1][i][j] + currents[1][i][j+1])/2) / (k[i] * numHistories * meshSize[geom[j]]))
                centerCurrents[2][i][j] = (((currents[2][i][j] + currents[2][i][j+1])/2) / (k[i] * numHistories *  meshSize[geom[j]]))


            # Recalculate fission source for next generation
            F_i = np.empty(len(geom))
            for j in range(len(geom)):
                if geom[j] == "W":
                    F_i[j] = 0
                elif geom[j] == "U":
                    F_i[j] = ((xsDict["U"]["Nu_f2"] * xsDict["U"]["Sigma_f2"] ) * flux2[i][j]) + ((xsDict["U"]["Nu_f1"] * xsDict["U"]["Sigma_f1"] ) * flux1[i][j])
                elif geom[j] == "M":
                    F_i[j] = ((xsDict["M"]["Nu_f2"] * xsDict["M"]["Sigma_f2"])) * flux2[i][j] + ((xsDict["M"]["Nu_f1"] * xsDict["M"]["Sigma_f1"]) * flux1[i][j])

            F_sum = 0
            for j in range(len(F_i)):
                F_sum += F_i[j] * meshSize[geom[j]]
            k[i+1] *= (k[i] * F_sum)
        
        runTime = time.time() - startTime
        print(f"Time elapsed: {runTime:.2f} seconds")

        fundamentalFlux1 = np.average(flux1[skipGenerations:,:],axis=0)
        fundamentalFlux1_std = np.std(flux1[skipGenerations:,:],axis=0)
        fundamentalFlux2 = np.average(flux2[skipGenerations:,:],axis=0)
        fundamentalFlux2_std = np.std(flux2[skipGenerations:,:],axis=0)
        fundamentalCurrent1 = np.average(centerCurrents[1][skipGenerations:,:],axis=0)
        fundamentalCurrent1_std = np.std(centerCurrents[1][skipGenerations:,:],axis=0)
        fundamentalCurrent2 = np.average(centerCurrents[2][skipGenerations:,:],axis=0)
        fundamentalCurrent2_std = np.std(centerCurrents[2][skipGenerations:,:],axis=0)
        fundamentalK = np.average(k[skipGenerations:])
        fundamentalK_std = np.std(k[skipGenerations:])

        # Find fuel bounds for plotting
        fuelBounds = []
        for i in range(len(geom)-1):
            if (geom[i] == "U" or geom[i] == "M") and geom[i+1] == "W":
                fuelBounds.append(cellRightBound(i))
            elif (geom[i+1] == "U" or geom[i+1] == "M") and geom[i] == "W":
                fuelBounds.append(cellRightBound(i))

        cellCenters = []
        for i in range(len(geom)):
            cellCenters.append(cellCenter(i))

        avgFlux1Assem1 = np.average(fundamentalFlux1[fuelCells[0:int(len(fuelCells)/2)]])
        avgFlux1Assem2 = np.average(fundamentalFlux1[fuelCells[int(len(fuelCells)/2):]])
        avgFlux2Assem1 = np.average(fundamentalFlux2[fuelCells[0:int(len(fuelCells)/2)]])
        avgFlux2Assem2 = np.average(fundamentalFlux2[fuelCells[int(len(fuelCells)/2):]])

        rxnRate1Assem1 = avgFlux1Assem1 * xsDict[geom[fuelCells[0]]]["Sigma_f1"]
        rxnRate1Assem2 = avgFlux1Assem2 * xsDict[geom[fuelCells[-1]]]["Sigma_f1"]
        rxnRate2Assem1 = avgFlux2Assem1 * xsDict[geom[fuelCells[0]]]["Sigma_f2"]
        rxnRate2Assem2 = avgFlux2Assem2 * xsDict[geom[fuelCells[-1]]]["Sigma_f2"]

        energyPerFission = 200 * 1.6022E-13 # Joules

        energyRelease = ((rxnRate1Assem1 + rxnRate1Assem2 + rxnRate2Assem1 + rxnRate2Assem2) * energyPerFission) / 1E6 # MW

        powerFactor = power / energyRelease

        print(f"Fundamental k: {fundamentalK:.5f}")
        print(f"Error: {fundamentalK_std/fundamentalK*100:.2f}%")

        plt.figure(figsize=(23,10)) 
        plt.errorbar(x=fuelBounds, y=np.zeros(len(fuelBounds)), linestyle='', yerr=(np.max(fundamentalFlux1)), color="g", alpha=0.3) # Fuel boundaries
        plt.errorbar(x=[0,L_geom], y=np.zeros(2), linestyle='', yerr=(np.max(fundamentalFlux1)), color="k", alpha=0.4) # Problem bounds
        plt.plot(cellCenters, fundamentalFlux1, label="Group 1 Flux", c="b")
        plt.scatter(cellCenters, fundamentalFlux1, c="b",s=0.5)  
        plt.plot(cellCenters, fundamentalFlux2, label="Group 2 Flux", c="orange")
        plt.scatter(cellCenters, fundamentalFlux2, c="orange",s=0.5) 
        plt.legend()
        plt.ylim(0, np.max(fundamentalFlux1))
        plt.xlabel("Position (cm)")
        plt.ylabel("Neutron Flux (1 / cm^2 - s)")
        plt.show()

        plt.figure(figsize=(23,10)) 
        plt.errorbar(x=fuelBounds, y=np.zeros(len(fuelBounds)), linestyle='', yerr=(np.max(fundamentalFlux1)*powerFactor), color="g", alpha=0.3) # Fuel boundaries
        plt.errorbar(x=[0,L_geom], y=np.zeros(2), linestyle='', yerr=(np.max(fundamentalFlux1)*powerFactor), color="k", alpha=0.4) # Problem bounds
        plt.plot(cellCenters, fundamentalFlux1 * powerFactor, label="Group 1 Flux", c="b")
        plt.scatter(cellCenters, fundamentalFlux1 * powerFactor, c="b",s=0.5)  
        plt.plot(cellCenters, fundamentalFlux2 * powerFactor, label="Group 2 Flux", c="orange")
        plt.scatter(cellCenters, fundamentalFlux2 * powerFactor, c="orange",s=0.5) 
        plt.legend()
        plt.ylim(0, np.max(fundamentalFlux1 * powerFactor))
        plt.xlabel("Position (cm)")
        plt.ylabel("Neutron Flux (1 / cm^2 - s)")
        plt.show()

        plt.figure(figsize=(23,10)) 
        plt.errorbar(x=fuelBounds, y=np.zeros(len(fuelBounds)), linestyle='', yerr=(np.max(fundamentalCurrent1)), color="g", alpha=0.3) # Fuel boundaries
        plt.errorbar(x=[0,L_geom], y=np.zeros(2), linestyle='', yerr=(np.max(fundamentalCurrent1)), color="k", alpha=0.4) # Problem bounds
        plt.plot(cellCenters, fundamentalCurrent1, label="Group 1 Current")
        plt.scatter(cellCenters, fundamentalCurrent1, c="b",s=0.5)
        plt.plot(cellCenters, fundamentalCurrent2, label="Group 2 Current")
        plt.scatter(cellCenters, fundamentalCurrent2, c="orange",s=0.5)
        # plt.errorbar(starts, np.zeros(len(starts)), linestyle="",xerr=0,yerr=0.01, c="r", label="Start Positions")
        plt.legend()
        plt.ylim(np.min(fundamentalCurrent1), np.max(fundamentalCurrent1))
        plt.xlabel("Position (cm)")
        plt.ylabel("Neutron Current")
        plt.show()

        # allBounds = list(set(leftBounds + rightBounds))
        # plt.figure(figsize=(23,10)) 
        # plt.errorbar(x=fuelBounds, y=np.zeros(len(fuelBounds)), linestyle='', yerr=(np.max(fundamentalCurrent1)), color="g", alpha=0.3) # Fuel boundaries
        # plt.errorbar(x=[0,L_geom], y=np.zeros(2), linestyle='', yerr=(np.max(fundamentalCurrent1)), color="k", alpha=0.4) # Problem bounds
        # #plt.plot(allBounds, currents[1][0], label="Group 1 Current")
        # plt.scatter(allBounds, currents[1][0], c="b")
        # #plt.plot(allBounds, currents[2][0], label="Group 2 Current")
        # plt.scatter(allBounds, currents[2][0], c="orange")
        # plt.errorbar(starts, np.zeros(len(starts)), linestyle="",xerr=0,yerr=0.01, c="r", label="Start Positions")
        # plt.legend()
        # #plt.ylim(np.min(fundamentalCurrent1), np.max(fundamentalCurrent1))
        # plt.show()

        plt.figure(figsize=(23,10)) 
        plt.plot(range(numGenerations+1), np.ones(numGenerations+1)*fundamentalK, c="k", label="Fundamental k")
        plt.scatter(range(skipGenerations), k[0:skipGenerations], label="Multiplication Factor (skipped)", marker="X", c="r", s=150.0)
        plt.scatter(range(skipGenerations, numGenerations+1), k[skipGenerations:], label="Multiplication Factor", c="b", s=150.0)
        plt.legend()
        plt.xlabel("Generation")
        plt.ylabel("Multiplication Factor")
        plt.show()
    
    if runFD:
        N = len(geom)

        # Create array for d values
        d_1 = np.empty(N)
        d_2 = np.empty(N)

        for i in range(N):
            d_1[i] = xsDict[geom[i]]["D_1"] / meshSize[geom[i]]
            d_2[i] = xsDict[geom[i]]["D_2"] / meshSize[geom[i]]

        # Build coefficient matrix
        coeffMatrix_1 = np.zeros((N,N))
        coeffMatrix_2 = np.zeros((N,N))

        d_1kk = lambda i1,i2: (2 * d_1[i1] * d_1[i2]) / (d_1[i1] + d_1[i2])
        d_2kk = lambda i1,i2: (2 * d_2[i1] * d_2[i2]) / (d_2[i1] + d_2[i2])

        # Manual creation at problem boundaries
        # Left bound
        if geometry[0] == "V" or geometry[0] == "W":
            beta_left_1 = 1 / (1 + (1 / (4 * d_1[0])))
            beta_left_2 = 1 / (1 + (1 / (4 * d_2[0])))

            coeffMatrix_1[0][0] = 2 * d_1[0] * (1 - beta_left_1) + meshSize[geom[0]] * xsDict[geom[0]]["Sigma_r1"] + d_1kk(0,1)
            coeffMatrix_1[0][1] = -d_1kk(0,1)

            coeffMatrix_2[0][0] = 2 * d_2[0] * (1 - beta_left_2) + meshSize[geom[0]] * xsDict[geom[0]]["Sigma_r1"] + d_2kk(0,1)
            coeffMatrix_2[0][1] = -d_2kk(0,1)
        else:
            beta_left_1 = 1
            beta_left_2 = 1

            coeffMatrix_1[0][0] = 2 * d_1[0] * (1 - beta_left_1) + meshSize[geom[0]] * xsDict[geom[0]]["Sigma_r1"] + d_1kk(0,1)
            coeffMatrix_1[0][1] = -d_1kk(0,1)

            coeffMatrix_2[0][0] = 2 * d_2[0] * (1 - beta_left_2) + meshSize[geom[0]] * xsDict[geom[0]]["Sigma_r1"] + d_2kk(0,1)
            coeffMatrix_2[0][1] = -d_2kk(0,1)
        
        # Right bound
        if geometry[-1] == "V" or geometry[-1] == "W":
            beta_left_1 = 1 / (1 + (1 / (4 * d_1[-1])))
            beta_left_2 = 1 / (1 + (1 / (4 * d_2[-1])))

            coeffMatrix_1[-1][-1] = 2 * d_1[-1] * (1 - beta_left_1) + meshSize[geom[-1]] * xsDict[geom[-1]]["Sigma_r1"] + d_1kk(-1,-2)
            coeffMatrix_1[-1][-2] = -d_1kk(-1,-2)

            coeffMatrix_2[-1][-1] = 2 * d_2[-1] * (1 - beta_left_2) + meshSize[geom[-1]] * xsDict[geom[-1]]["Sigma_r1"] + d_2kk(-1,-2)
            coeffMatrix_2[-1][-2] = -d_2kk(-1,-2)

        else:
            beta_left_1 = 1
            beta_left_2 = 1

            coeffMatrix_1[-1][-1] = 2 * d_1[-1] * (1 - beta_left_1) + meshSize[geom[-1]] * xsDict[geom[0]]["Sigma_r1"] + d_1kk(-1,-2)
            coeffMatrix_1[-1][-2] = -d_1kk(-1,-2)

            coeffMatrix_2[-1][-1] = 2 * d_2[-1] * (1 - beta_left_2) + meshSize[geom[-1]] * xsDict[geom[0]]["Sigma_r1"] + d_2kk(-1,-2)
            coeffMatrix_2[-1][-2] = -d_2kk(-1,-2)
        
        # Populate matrix
        for i in range(1,N-1):
            coeffMatrix_1[i][i-1] = -d_1kk(i,i-1)
            coeffMatrix_1[i][i] = d_1kk(i,i-1) + d_1kk(i,i+1) + meshSize[geom[i]] * xsDict[geom[i]]["Sigma_r1"]
            coeffMatrix_1[i][i+1] = -d_1kk(i,i+1)

            coeffMatrix_2[i][i-1] = -d_2kk(i,i-1)
            coeffMatrix_2[i][i] = d_2kk(i,i-1) + d_2kk(i,i+1) + meshSize[geom[i]] * xsDict[geom[i]]["Sigma_r2"]
            coeffMatrix_2[i][i+1] = -d_2kk(i,i+1)
        
        # Initialize values

        flux1 = np.ones(N)
        flux2 = np.ones(N)

        k = 1.0

        Q = np.ones(N)
        for i in range(N):
            Q[i] *= (meshSize[geom[i]] * (xsDict[geom[i]]["Nu_f1"] * xsDict[geom[i]]["Sigma_f1"] * flux1[i] + xsDict[geom[i]]["Nu_f2"] * xsDict[geom[i]]["Sigma_f2"] * flux2[i]))

        # TODO: Put this in the input file
        epsilon_k = 1E-5
        epsilon_flux1 = 1E-3
        epsilon_flux2 = 1E-3

        flux1_diff = 1.0
        flux2_diff = 1.0
        k_diff = 1.0

        # Enter iteration loop for solving
        while k_diff > epsilon_k or flux1_diff > epsilon_flux1 or flux2_diff > epsilon_flux2:
            rhs_1 = Q / k

            flux1_new = np.linalg.solve(coeffMatrix_1, rhs_1)

            rhs_2 = np.zeros(N)
            for i in range(N):
                rhs_2[i] += ( meshSize[geom[i]] * xsDict[geom[i]]["Sigma_s12"] * flux1_new[i])
            
            flux2_new = np.linalg.solve(coeffMatrix_2, rhs_2)

            Q_new = np.ones(N)
            for i in range(N):
                Q_new[i] *= (meshSize[geom[i]] * (xsDict[geom[i]]["Nu_f1"] * xsDict[geom[i]]["Sigma_f1"] * flux1_new[i] + xsDict[geom[i]]["Nu_f2"] * xsDict[geom[i]]["Sigma_f2"] * flux2_new[i]))

            k_new = np.sum(Q_new) / (np.sum(Q) / k)

            k_diff = abs(k_new - k)
            flux1_diff = np.max(abs(flux1_new - flux1))
            flux2_diff = np.max(abs(flux2_new - flux2))

            flux1 = flux1_new
            flux2 = flux2_new
            k = k_new

            print(k)



if __name__ == "__main__":
    main("inp2.txt")