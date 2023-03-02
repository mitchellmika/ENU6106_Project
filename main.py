import csv
import numpy as np
import matplotlib.pyplot as plt
import math

def main(inputFile):
    # Read input file
    with open(inputFile, mode="r") as file:
        readlines = file.readlines()
        method = readlines[0]
        meshSize = float(readlines[1])
        geometry = readlines[2]
        dataSet = int(readlines[3])
        power = float(readlines[4])
        numHistories = int(readlines[5])
        numGenerations = int(readlines[6])

    # Input checks
    # TODO: Check if mesh size fits, try to bump to closest value that does fit
    
    # Read data file
    with open(f"xs_set{dataSet}.csv", mode="r") as file:
        csvFile = csv.reader(file)
        xsDict = {}
        for line in csvFile:
            xsDict[line[0]] = {
                "Sigma_tr1":float(line[1]),
                "Sigma_s11":float(line[2]),
                "Sigma_s12":float(line[3]),
                "Sigma_a1":float(line[4]),
                "Sigma_f1":float(line[5]),
                "Sigma_c1":float(line[4]) - float(line[5]),
                "Sigma_t1":float(line[2]) + float(line[3]) + float(line[4]),
                "Capture_1":(float(line[4]) - float(line[5])) / (float(line[2]) + float(line[3]) + float(line[4])), # Upper cdf bound for capture interaction in g1
                "Fission_1":float(line[4]) / (float(line[2]) + float(line[3]) + float(line[4])), # Upper cdf bound for fission in g1
                "InScatter_1":(float(line[4]) + float(line[2])) / (float(line[2]) + float(line[3]) + float(line[4])), # Upper cdf bound for inscatter in g1
                "Nu_f1":float(line[6]),
                "Chi_1":float(line[7]),
                "Sigma_tr2":float(line[8]),
                "Sigma_s22":float(line[9]),
                "Sigma_s21":float(line[10]),
                "Sigma_a2":float(line[11]),
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
    D_fuel = 1.1058
    P = 1.4385
    L_assem = 24.4545
    D_water = P - D_fuel
    waterMeshesPerPin = int(D_water / meshSize)
    fuelMeshesPerPin = int(D_fuel/meshSize)

    geom = []
    for mat in geometry[1:3]:
        for i in range(17):
            geom.extend("W" * int(waterMeshesPerPin/2) )
            geom.extend(mat * fuelMeshesPerPin)
            geom.extend("W" * int(waterMeshesPerPin/2) )

    
    # Define fission distribution
    fuelCells = np.where(np.array(geom) != "W")[0]
    UCells = np.where(np.array(geom) == "U")[0]
    MCells = np.where(np.array(geom) == "M")[0]
    numUCells = len(UCells)
    numMCells = len(MCells)

    normalizingFactor = numUCells * xsDict["U"]["Sigma_f2"] + numMCells * xsDict["M"]["Sigma_f2"]
    F_U = xsDict["U"]["Sigma_f2"] / normalizingFactor
    F_M = xsDict["M"]["Sigma_f2"] / normalizingFactor
    
    xsDict["M"]["F_i"] = F_M
    xsDict["U"]["F_i"] = F_U
    xsDict["W"]["F_i"] = 0        

    # Functions for finding left and right bounds given cell index
    cellRightBound = lambda index: (index + 1) * meshSize
    cellLeftBound = lambda index: index * meshSize
    cellCenter = lambda index: (index + 0.5) * meshSize

    # Function for find index based on position
    posToIndex = lambda pos: int(pos / meshSize)

    # Zeros arrays for track length tallying
    trackLengths = {1:np.zeros(len(geom)), 2:np.zeros(len(geom))}


    #TODO: Refactor so that main if statement checks cross boundary or not, and then split up the right or left logic at each next step
    # TODO: Might be slower but will make debugging much easier
    # Begin generation loop
    starts = []
    for i in range(numGenerations):
        # Begin history loop
        for j in range(numHistories):
            # Determine cell start index
            cellIndex = -1
            startCellRandNum = np.random.random(1)[0]
            for cell in fuelCells:
                startCellRandNum -= xsDict[geom[cell]]["F_i"]
                if startCellRandNum < 0:
                    cellIndex = cell
                    starts.append(cellIndex)
                    break
            
            # Determine start position
            pos = np.random.uniform(cellLeftBound(cellIndex), cellRightBound(cellIndex))
            energyGroup = 1

            absorbed = False
            resampleDir = True
            while not absorbed:
                material = geom[cellIndex]
                if resampleDir:
                    squiggly = np.random.random(1)[0]
                    mu = 2*squiggly - 1

                travelDistance = (-math.log(squiggly) / xsDict[material][f"Sigma_t{energyGroup}"]) * mu

                # Update position and cell index (assuming no mat or problem boundary crossed, will check later)
                newPos = travelDistance + pos
                newCellIndex = posToIndex(newPos)

                ##########################################################################################################################################################
                # Check for material crosing along the way
                # If material changes, need to add track lengths and resample travel distance at material boundary
                #########################################################################################################################################################
                matChange = False

                # For right to left
                if travelDistance < 0:
                    edgeCell = 0 if newCellIndex < 0 else newCellIndex # Use to handle cases where particle crosses problem boundary

                    for cell in range(cellIndex,edgeCell, -1): # iterate from travelled distance from right to left before original cell to find where material change occurs
                        if geom[cell] is not geom[cellIndex]:
                            newCellIndex = cell
                            matChange = True
                            break
                    
                    if matChange:
                        trackLengths[energyGroup][newCellIndex+1:cellIndex] += meshSize # Add track length to all cells until material change
                        trackLengths[energyGroup][cellIndex] += pos - cellLeftBound(cellIndex) # Add track length to original cell
                        pos = cellRightBound(newCellIndex) # Move neutron to where material boundary was crossed  (right edge of cell bc moving right to left)
                        cellIndex = newCellIndex
                        material = geom[cellIndex] # Update material
                        resampleDir = False
                        continue

                # For left to right
                else:
                    edgeCell = len(geom)-1 if newCellIndex > len(geom)-1 else newCellIndex # Use to handle cases where particle crosses problem boundary

                    for cell in range(cellIndex,edgeCell+1): # iterate from travelled distance from right to left before original cell to find where material change occurs
                        if geom[cell] is not geom[cellIndex]:
                            newCellIndex = cell
                            matChange = True
                            break
                    
                    if matChange:
                        trackLengths[energyGroup][cellIndex+1:newCellIndex] += meshSize # Add track length to all cells until material change
                        trackLengths[energyGroup][cellIndex] += cellRightBound(cellIndex) - pos # Add track length to original cell
                        pos = cellLeftBound(newCellIndex) # Move neutron to boundary (left bound bc traveling left to right)
                        cellIndex = newCellIndex
                        material = geom[cellIndex] # Update material
                        resampleDir = False
                        continue
                    
                #######################################################################################################################################################
                # Check for problem boundaries
                #######################################################################################################################################################
                if newPos < 0 and geometry[0] == "V":
                    # Particle is lost to vacuum on left side
                    trackLengths[energyGroup][cellIndex] += pos - cellLeftBound(cellIndex)
                    if cellIndex != 0: 
                        trackLengths[energyGroup][0:cellIndex] += meshSize
                    absorbed = True
                    continue

                elif newPos < 0 and geometry[0] == "I":
                    # Particle is reflected on left side
                    # TODO: Calculate distance that particle "bounces" off the boundary
                    trackLengths[energyGroup][cellIndex] += abs(travelDistance)

                elif newPos > (meshSize * len(geom)) and geometry[3] == "V":
                    # Particle is lost to vacuum on right side
                    trackLengths[energyGroup][cellIndex] += cellRightBound(cellIndex) - pos # Add track length to start cell
                    if cellIndex != len(geom)-1: 
                        trackLengths[energyGroup][cellIndex+1:len(geom)-1] += meshSize # Add track length to other cells if start cell was not at boundary
                    absorbed = True
                    continue

                elif newPos > (meshSize * len(geom)) and geometry[3] == "I":
                    # Particle is reflected on right side
                    # TODO: Reflection distance, need to check boundary crossing again?
                    trackLengths[energyGroup][cellIndex] += abs(travelDistance)
                
                ##################################################################################################################################################
                # No material changes, no problem boundaries crossed
                # Compute tally lengths and sample interaction
                #################################################################################################################################################

                if newCellIndex != cellIndex and travelDistance < 0:
                    trackLengths[energyGroup][cellIndex] += pos - cellLeftBound(cellIndex) # Add track length to original cell
                    trackLengths[energyGroup][newCellIndex+1:cellIndex] += meshSize # Add track length to cells along the way
                    trackLengths[energyGroup][newCellIndex] += cellRightBound(newCellIndex) - newPos # Add track length to new cell location
                elif newCellIndex != cellIndex and travelDistance > 0:
                    trackLengths[energyGroup][cellIndex] += cellRightBound(cellIndex) - pos # Add track length to original cell
                    trackLengths[energyGroup][cellIndex+1:newCellIndex] += meshSize # Add track length to cells along the way
                    trackLengths[energyGroup][newCellIndex] += newPos - cellLeftBound(newCellIndex) # Add track length to new cell location
                else:
                    trackLengths[energyGroup][cellIndex] += abs(travelDistance)

                pos = newPos
                cellIndex = newCellIndex

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
                
                # print(f"Neutron {j} in cell {cellIndex} / {len(geom)}, at position {pos:.7f} / {meshSize * len(geom):.7f}")
                resampleDir = True
                    
    fuelBounds = []
    for i in range(len(geom)-1):
        if (geom[i] == "U" or geom[i] == "M") and geom[i+1] == "W":
            fuelBounds.append(i)
        elif (geom[i+1] == "U" or geom[i+1] == "M") and geom[i] == "W":
            fuelBounds.append(i+1)

    plt.figure(figsize=(23,13)) 
    plt.errorbar(x=fuelBounds, y=np.ones(len(fuelBounds))*np.mean(trackLengths[1]), linestyle='', yerr=(np.mean(trackLengths[1])), color="g", alpha=0.3) # Fuel boundaries
    plt.errorbar(x=[0,len(geom)-1], y=np.ones(2)*np.mean(trackLengths[1]), linestyle='', yerr=(np.mean(trackLengths[1])), color="k", alpha=0.4)
    plt.plot(range(len(geom)), trackLengths[1], label="Track Lengths")
    plt.scatter(starts, np.zeros(len(starts)), c="r", s=0.5, label="Start Positions")
    plt.legend()
    plt.show()



    


if __name__ == "__main__":
    main("inp.txt")