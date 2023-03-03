import csv
import numpy as np
import matplotlib.pyplot as plt
import math

def AddCurrent(currents, energyGroup, startCell, endCell):
    # Left to right motion
    if startCell < endCell:
        firstEdge = startCell + 1
        lastEdge = endCell

        currents[energyGroup][firstEdge:lastEdge+1] += mu
    # Right to left motion
    else:
        pass

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

    normalizingFactor = (numUCells * (xsDict["U"]["Sigma_f2"]* xsDict["U"]["Nu_f2"] + xsDict["U"]["Sigma_f1"]* xsDict["U"]["Nu_f1"])) + (numMCells * (xsDict["M"]["Sigma_f2"]* xsDict["M"]["Nu_f2"] + xsDict["M"]["Sigma_f1"]* xsDict["M"]["Nu_f1"]))
    F_U = (xsDict["U"]["Sigma_f2"]* xsDict["U"]["Nu_f2"] + xsDict["U"]["Sigma_f1"]* xsDict["U"]["Nu_f1"]) / normalizingFactor
    F_M = (xsDict["M"]["Sigma_f2"]* xsDict["M"]["Nu_f2"] + xsDict["M"]["Sigma_f1"]* xsDict["M"]["Nu_f1"]) / normalizingFactor
    
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

    # Functions for finding left and right bounds given cell index
    cellRightBound = lambda index: (index + 1) * meshSize
    cellLeftBound = lambda index: index * meshSize
    cellCenter = lambda index: (index + 0.5) * meshSize

    # Function for find index based on position
    posToIndex = lambda pos: int(pos / meshSize)

    # Zeros arrays for track length tallying
    trackLengths = {1:np.zeros(len(geom)), 2:np.zeros(len(geom))}
    currents = {1:np.zeros(len(geom)+1), 2:np.zeros(len(geom)+1)}
    currents = {1:np.zeros(len(geom)+1), 2:np.zeros(len(geom)+1)}

    # Begin generation loop
    # TODO: Current tracking for infinite lattice condition
    starts = []
    leftDir = []
    rightDir =[]
    k = 1.0

    # FIXME: Current at right edge is not dropping to zero. Maybe indexing error?
    for i in range(numGenerations):
        # Begin history loop
        for j in range(numHistories):
            # Determine cell start index
            cellIndex = -1
            startCellRandNum = np.random.random(1)[0]
            for cell in fuelCells:
                startCellRandNum -= F[cell]
                if startCellRandNum < 0:
                    cellIndex = cell
                    starts.append(cellIndex)
                    break
            
            # Determine start position
            pos = np.random.uniform(cellLeftBound(cellIndex), cellRightBound(cellIndex))
            energyGroup = 1

            absorbed = False
            resampleDir = True
            resampleDist = True
            while not absorbed:
                #print(f"Neutron {j} in cell {cellIndex} / {len(geom)}, at position {pos:.7f} / {meshSize * len(geom):.7f}")

                material = geom[cellIndex]

                # FIXME: Left dir magnitude still seems much bigger, almost double
                squiggly = np.random.random(1)[0]
                if resampleDir:
                    mu = 2*squiggly - 1

                if resampleDist:
                    travelDistance = (-math.log(squiggly) / xsDict[material][f"Sigma_t{energyGroup}"]) * mu
                if travelDistance < 0:
                    leftDir.append(travelDistance)
                else:
                    rightDir.append(travelDistance)

                # Update position and cell index (assuming no mat or problem boundary crossed, will check later)
                newPos = travelDistance + pos
                newCellIndex = posToIndex(newPos)

                ##########################################################################################################################################################
                # Check for material crosing along the way
                # If material changes, need to add track lengths and resample travel distance at material boundary
                #########################################################################################################################################################
                # For right to left
                if travelDistance < 0:
                    matChange = False
                    edgeCell = 0 if newCellIndex < 0 else newCellIndex # Use to handle cases where particle crosses problem boundary

                    for cell in range(cellIndex,edgeCell, -1): # iterate from travelled distance from right to left before original cell to find where material change occurs
                        if geom[cell] is not geom[cellIndex]: # Enter if material change found along path
                            newCellIndex = cell

                            # Update track lengths
                            trackLengths[energyGroup][newCellIndex+1:cellIndex] += meshSize # Add track length to all cells until material change
                            trackLengths[energyGroup][cellIndex] += (pos - cellLeftBound(cellIndex)) # Add track length to original cell
                            pos = cellRightBound(newCellIndex) # Move neutron to where material boundary was crossed  (right edge of cell bc moving right to left)

                            # Update currents
                            firstEdge = cellIndex
                            lastEdge = newCellIndex + 1
                            currents[energyGroup][lastEdge:firstEdge+1] += mu

                            cellIndex = newCellIndex
                            resampleDir = False
                            resampleDist = True
                            matChange=True
                            break
                        
                    if matChange:
                        continue

                # For left to right
                else:
                    matChange = False
                    edgeCell = len(geom)-1 if newCellIndex > len(geom)-1 else newCellIndex # Use to handle cases where particle crosses problem boundary

                    for cell in range(cellIndex,edgeCell+1): # iterate from travelled distance from right to left before original cell to find where material change occurs
                        if geom[cell] is not geom[cellIndex]: # Enter if material change found along path
                            newCellIndex = cell

                            # Update track lengths
                            trackLengths[energyGroup][cellIndex+1:newCellIndex] += meshSize # Add track length to all cells until material change
                            trackLengths[energyGroup][cellIndex] += (cellRightBound(cellIndex) - pos) # Add track length to original cell
                            pos = cellLeftBound(newCellIndex) # Move neutron to boundary (left bound bc traveling left to right)

                            # Update currents
                            firstEdge = cellIndex + 1
                            lastEdge = newCellIndex
                            currents[energyGroup][firstEdge:lastEdge+1] += mu

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
                    trackLengths[energyGroup][cellIndex] += (pos - cellLeftBound(cellIndex))
                    if cellIndex != 0: 
                        trackLengths[energyGroup][0:cellIndex] += meshSize
                    absorbed = True

                    # Update currents
                    firstEdge = cellIndex
                    lastEdge = newCellIndex + 1
                    currents[energyGroup][lastEdge:firstEdge+1] += mu
                    continue

                elif newPos < 0 and geometry[0] == "I":
                    # Particle is reflected on left side
                    trackLengths[energyGroup][cellIndex] += (2 * (pos - cellLeftBound(cellIndex)))
                    if cellIndex != 0: 
                        trackLengths[energyGroup][0:cellIndex] += (2 * meshSize)
                    
                    # Update currents
                    firstEdge = cellIndex
                    lastEdge = newCellIndex + 1
                    currents[energyGroup][lastEdge:firstEdge+1] += mu

                    travelDistance += pos
                    travelDistance *= -1 # Need to flip travel distance and direction
                    mu *= -1
                    resampleDir = False
                    resampleDist = False
                    continue

                elif newPos > (meshSize * len(geom)) and geometry[3] == "V":
                    # Particle is lost to vacuum on right side
                    trackLengths[energyGroup][cellIndex] += (cellRightBound(cellIndex) - pos) # Add track length to start cell
                    if cellIndex != len(geom)-1: 
                        trackLengths[energyGroup][cellIndex+1:len(geom)-1] += meshSize # Add track length to other cells if start cell was not at boundary
                    
                    # Update currents
                    firstEdge = cellIndex + 1
                    lastEdge = newCellIndex
                    currents[energyGroup][firstEdge:lastEdge+1] += mu
                    absorbed = True
                    continue

                elif newPos > (meshSize * len(geom)) and geometry[3] == "I":
                    # Particle is reflected on right side
                    trackLengths[energyGroup][cellIndex] += (2 * (cellRightBound(cellIndex) - pos))
                    if cellIndex != len(geom)-1: 
                        trackLengths[energyGroup][cellIndex+1:len(geom)-1] += (2 * meshSize) # Add track length to other cells if start cell was not at boundary

                    # Update currents
                    firstEdge = cellIndex + 1
                    lastEdge = newCellIndex
                    currents[energyGroup][firstEdge:lastEdge+1] += mu

                    travelDistance -= (cellRightBound(-1) - pos)
                    travelDistance *= -1 # Need to flip travel distance and direction
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
                    trackLengths[energyGroup][cellIndex] += (pos - cellLeftBound(cellIndex)) # Add track length to original cell
                    trackLengths[energyGroup][newCellIndex+1:cellIndex] += meshSize # Add track length to cells along the way
                    trackLengths[energyGroup][newCellIndex] += (cellRightBound(newCellIndex) - newPos) # Add track length to new cell location

                    # Update currents
                    firstEdge = cellIndex
                    lastEdge = newCellIndex + 1
                    currents[energyGroup][lastEdge:firstEdge+1] += mu

                # Left to right
                elif newCellIndex != cellIndex and travelDistance > 0:
                    # Update track lengths
                    trackLengths[energyGroup][cellIndex] += (cellRightBound(cellIndex) - pos) # Add track length to original cell
                    trackLengths[energyGroup][cellIndex+1:newCellIndex] += meshSize # Add track length to cells along the way
                    trackLengths[energyGroup][newCellIndex] += (newPos - cellLeftBound(newCellIndex)) # Add track length to new cell location

                    # Update currents
                    firstEdge = cellIndex + 1
                    lastEdge = newCellIndex
                    currents[energyGroup][firstEdge:lastEdge+1] += mu
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
                
                resampleDir = True
                resampleDist = True
        
        pass # This is where history loop ends
        flux1 = trackLengths[1] / (k * meshSize * numHistories)
        flux2 = trackLengths[2] / (k * meshSize * numHistories)
        
        # Recalculate fission source for next generation
        for l in range(len(geom)):
            if geom[l] == "W":
                F[l] = 0
            elif geom[l] == "U":
                F[l] = (xsDict["U"]["Nu_f1"] * xsDict["U"]["Sigma_f1"] * flux1[l]) + (xsDict["U"]["Nu_f2"] * xsDict["U"]["Sigma_f2"] * flux2[l])
            elif geom[l] == "M":
                F[l] = (xsDict["M"]["Nu_f1"] * xsDict["M"]["Sigma_f1"] * flux1[l]) + (xsDict["M"]["Nu_f2"] * xsDict["M"]["Sigma_f2"] * flux2[l])

        k *= (meshSize * np.sum(F))

        normalizingFactor = np.sum(F)
        F /= normalizingFactor

        print(k)
    
    # Calculate currents in centers of mesh
    centerCurrents = {1:np.empty(len(currents[1])-1), 2:np.empty(len(currents[1])-1)}
    for j in range(len(centerCurrents[1])):
        centerCurrents[1][j] = ((currents[1][j] + currents[1][j+1]) / (numHistories * meshSize))
        centerCurrents[2][j] = ((currents[2][j] + currents[2][j+1]) / (numHistories * meshSize))

    # Find fuel bounds for plotting
    fuelBounds = []
    for i in range(len(geom)-1):
        if (geom[i] == "U" or geom[i] == "M") and geom[i+1] == "W":
            fuelBounds.append(i)
        elif (geom[i+1] == "U" or geom[i+1] == "M") and geom[i] == "W":
            fuelBounds.append(i+1)

    # print(np.mean(np.array(leftDir)))
    # print(np.mean(np.array(rightDir)))

    flux1 = trackLengths[1] / (meshSize * numHistories * numGenerations)
    flux2 = trackLengths[2] / (meshSize * numHistories * numGenerations)

    plt.figure(figsize=(23,12)) 
    plt.errorbar(x=fuelBounds, y=np.zeros(len(fuelBounds)), linestyle='', yerr=(np.max(flux1)), color="g", alpha=0.3) # Fuel boundaries
    plt.errorbar(x=[0,len(geom)-1], y=np.zeros(2)*np.max(flux1), linestyle='', yerr=(np.max(flux1)), color="k", alpha=0.4) # Problem bounds
    plt.plot(range(len(geom)), flux1, label="Group 1 Flux") 
    plt.plot(range(len(geom)), flux2, label="Group 2 Flux")
    plt.errorbar(starts, np.zeros(len(starts)), linestyle="",xerr=0,yerr=0.01, c="r", label="Start Positions")
    plt.legend()
    plt.ylim(0, np.max(flux1))
    plt.show()

    plt.figure(figsize=(23,12)) 
    plt.errorbar(x=fuelBounds, y=np.zeros(len(fuelBounds)), linestyle='', yerr=(np.max(flux1)), color="g", alpha=0.3) # Fuel boundaries
    plt.errorbar(x=[0,len(geom)-1], y=np.zeros(2)*np.max(flux1), linestyle='', yerr=(np.max(flux1)), color="k", alpha=0.4) # Problem bounds
    plt.plot(range(len(geom)), centerCurrents[1], label="Group 1 Current")
    plt.plot(range(len(geom)), centerCurrents[2], label="Group 2 Current")
    plt.errorbar(starts, np.zeros(len(starts)), linestyle="",xerr=0,yerr=0.01, c="r", label="Start Positions")
    plt.legend()
    plt.ylim(np.min(centerCurrents[1]), np.max(centerCurrents[1]))
    plt.show()



    


if __name__ == "__main__":
    main("inp.txt")