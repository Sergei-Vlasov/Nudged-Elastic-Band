
import numpy as np
import time

eps = 1e-8
springK = 10
delta = 1e-2
maxIterCount = 100000
maxstep = 0.01
theta = np.pi / 10000

g = 2.11
mB = 5.788381806638e-2


class Atom:
    def __init__(self, M, coordinates):
        self.M = M
        self.coordinates = coordinates
    
    def __str__(self):
        res = "Magnetic moment: " + str(self.M) + "\n" + \
            "Vector oordinates (x, y, z) of the magnetic moment: " + str(self.coordinates)
        return res

class Image:
    def __init__(self, listOfAtoms, axisAnisotropy, planeAnisotropy, exchange, field):
        self.size = len(listOfAtoms)
        self.listOfAtoms = listOfAtoms
        self.axisAnisotropy = axisAnisotropy
        self.planeAnisotropy = planeAnisotropy
        self.exchange = exchange
        self.field = field
        self.energy = self.calculateEnergy()
        self.tau = np.zeros((self.size, 3))
        self.force = np.zeros((self.size, 3))
        self.velocity = np.zeros((self.size, 3))
        self.previous = None
        self.next = None

    
    def calculateEnergy(self):
        energy = 0
        for i in range(self.size-1):
            energy += self.axisAnisotropy * self.listOfAtoms[i].M ** 2 * self.listOfAtoms[i].coordinates[0] ** 2 + self.planeAnisotropy * self.listOfAtoms[i].M ** 2 * (self.listOfAtoms[i].coordinates[1] ** 2 - self.listOfAtoms[i].coordinates[2] ** 2) + self.exchange * self.listOfAtoms[i].M * self.listOfAtoms[i + 1].M * np.dot(self.listOfAtoms[i].coordinates, self.listOfAtoms[i + 1].coordinates) - g * mB * self.listOfAtoms[i].M * np.dot(self.field, self.listOfAtoms[i].coordinates)
        return energy + self.axisAnisotropy * self.listOfAtoms[-1].M ** 2 * self.listOfAtoms[-1].coordinates[0] ** 2 + self.planeAnisotropy * self.listOfAtoms[-1].M ** 2 * (self.listOfAtoms[-1].coordinates[1] ** 2 - self.listOfAtoms[-1].coordinates[2] ** 2) - g * mB * self.listOfAtoms[-1].M * np.dot(self.field, self.listOfAtoms[-1].coordinates)
            
            
    def calculateXDerivative(self, num):
        deriv = 2 * self.axisAnisotropy * self.listOfAtoms[num].M ** 2 * self.listOfAtoms[num].coordinates[0] - g * mB * self.field[0] * self.listOfAtoms[num].M
        if num == 0:
            return deriv + self.exchange * self.listOfAtoms[num].M * self.listOfAtoms[num + 1].M * self.listOfAtoms[num + 1].coordinates[0]
        elif num == self.size - 1:
            return deriv + self.exchange * self.listOfAtoms[num].M * self.listOfAtoms[num - 1].M * self.listOfAtoms[num - 1].coordinates[0]
        else:
            return deriv + self.exchange * self.listOfAtoms[num].M * (self.listOfAtoms[num + 1].coordinates[0] * self.listOfAtoms[num + 1].M + self.listOfAtoms[num - 1].coordinates[0] * self.listOfAtoms[num - 1].M)


    def calculateYDerivative(self, num):
        deriv = 2 * self.planeAnisotropy * self.listOfAtoms[num].M ** 2 * self.listOfAtoms[num].coordinates[1] - g * mB * self.field[1] * self.listOfAtoms[num].M
        if num == 0:
            return deriv + self.exchange * self.listOfAtoms[num].M * self.listOfAtoms[num + 1].M * self.listOfAtoms[num + 1].coordinates[1]
        elif num == self.size - 1:
            return deriv + self.exchange * self.listOfAtoms[num].M * self.listOfAtoms[num - 1].M * self.listOfAtoms[num - 1].coordinates[1]
        else:
            return deriv + self.exchange * self.listOfAtoms[num].M * (self.listOfAtoms[num + 1].coordinates[1] * self.listOfAtoms[num + 1].M + self.listOfAtoms[num - 1].coordinates[1] * self.listOfAtoms[num - 1].M)
    
    def calculateZDerivative(self, num):
        deriv = -2 * self.planeAnisotropy * self.listOfAtoms[num].M ** 2 * self.listOfAtoms[num].coordinates[2] - g * mB * self.field[2] * self.listOfAtoms[num].M
        if num == 0:
            return deriv + self.exchange * self.listOfAtoms[num].M * self.listOfAtoms[num + 1].M * self.listOfAtoms[num + 1].coordinates[2]
        elif num == self.size - 1:
            return deriv + self.exchange * self.listOfAtoms[num].M * self.listOfAtoms[num - 1].M * self.listOfAtoms[num - 1].coordinates[2]
        else:
            return deriv + self.exchange * self.listOfAtoms[num].M * (self.listOfAtoms[num + 1].coordinates[2] * self.listOfAtoms[num + 1].M + self.listOfAtoms[num - 1].coordinates[2] * self.listOfAtoms[num - 1].M)


    def calculateGradient(self):
        return np.array([[-self.calculateXDerivative(num), -self.calculateYDerivative(num), -self.calculateZDerivative(num)] for num in range(self.size)])
    
    def calculateTau(self):
        atomsCount = self.size
        previous = self.previous
        next = self.next
        if self.energy > previous.energy and self.energy < next.energy:
            imageCur = np.array([atom.coordinates for atom in self.listOfAtoms]).reshape(1, 3 * atomsCount)
            imageNxt = np.array([atom.coordinates for atom in next.listOfAtoms]).reshape(1, 3 * atomsCount)
            temp = imageNxt - imageCur
        elif self.energy < previous.energy and self.energy > next.energy:
            imageCur = np.array([atom.coordinates for atom in self.listOfAtoms]).reshape(1, 3 * atomsCount)
            imagePrv = np.array([atom.coordinates for atom in previous.listOfAtoms]).reshape(1, 3 * atomsCount)
            temp = imageCur - imagePrv
        elif next.energy > previous.energy:
            imageCur = np.array([atom.coordinates for atom in self.listOfAtoms]).reshape(1, 3 * atomsCount)
            imageNxt = np.array([atom.coordinates for atom in next.listOfAtoms]).reshape(1, 3 * atomsCount)
            imagePrv = np.array([atom.coordinates for atom in previous.listOfAtoms]).reshape(1, 3 * atomsCount)
            temp = (imageNxt - imageCur) * max(np.abs(next.energy - self.energy), np.abs(previous.energy - self.energy)) + (imageCur - imagePrv) * min(np.abs(next.energy - self.energy), np.abs(previous.energy - self.energy))
        else:
            imageCur = np.array([atom.coordinates for atom in self.listOfAtoms]).reshape(1, 3 * atomsCount)
            imageNxt = np.array([atom.coordinates for atom in next.listOfAtoms]).reshape(1, 3 * atomsCount)
            imagePrv = np.array([atom.coordinates for atom in previous.listOfAtoms]).reshape(1, 3 * atomsCount)
            temp = (imageNxt - imageCur) * min(np.abs(next.energy - self.energy), np.abs(previous.energy - self.energy)) + (imageCur - imagePrv) * max(np.abs(next.energy - self.energy), np.abs(previous.energy - self.energy))
        m = imageCur.reshape(atomsCount, 3)
        temp = temp.reshape(atomsCount, 3)
        tau = np.array([temp[i] - np.dot(temp[i], m[i]) * m[i] for i in range(atomsCount)])
        return tau / np.linalg.norm(tau)
    
    def calculateTrueForce(self):
        atomsCount = self.size
        gradient = self.calculateGradient()
        tau = self.tau
        m = np.array([atom.coordinates for atom in self.listOfAtoms])
        temp = np.array(gradient - np.dot(gradient.reshape(3 * atomsCount), tau.reshape(3 * atomsCount)) * tau)
        force = np.array([temp[i] - np.dot(temp[i], m[i]) * m[i] for i in range(self.size)])
        return force
    
    def calculateSpringForce(self):
        tau = self.tau
        atomsCount = self.size
        previous = self.previous
        next = self.next
        imageCur = np.array([atom.coordinates for atom in self.listOfAtoms])
        imageNxt = np.array([atom.coordinates for atom in next.listOfAtoms])
        imagePrv = np.array([atom.coordinates for atom in previous.listOfAtoms])
        force = springK * (distance(imageNxt, imageCur) - distance(imageCur, imagePrv)) * tau
        return force
    
    def calculateForce(self):
        springForce = self.calculateSpringForce()
        trueForce = self.calculateTrueForce()
        self.force = springForce + trueForce
    
    def makeStep(self, method):
        if method == 'quick-min':
            for i in range(self.size):
                if np.dot(self.velocity[i], self.force[i]) < 0:
                    self.velocity[i] = [0, 0, 0]
                else:
                    self.velocity[i] += self.force[i] * delta
                    self.velocity[i] = np.dot(self.velocity[i], self.force[i]) * self.force[i] / (np.linalg.norm(self.force[i]) ** 2)
            step = self.velocity * delta + self.force * (delta ** 2)
        else:
            step = self.force * delta
        n = [np.cross(self.listOfAtoms[i].coordinates, -step[i]) for i in range(self.size)]
        n = list(map(lambda x: x / np.linalg.norm(x), n))
        for i in range(self.size):
            m = self.listOfAtoms[i].coordinates
            angle = np.arctan(np.linalg.norm(step[i]))
            if angle > theta: angle = theta
            self.listOfAtoms[i].coordinates = np.dot(m, rotationMatrix(n[i], angle))
        self.energy = self.calculateEnergy()
        self.tau = self.calculateTau()
        isConsist = True
        normOfForce = list(map(np.linalg.norm, self.force))
        for i in range(self.size):
            isConsist &= normOfForce[i] < eps
        return isConsist
    
    def minimization(self):
        itr = 0
        isConsist = False
        while not isConsist and itr < maxIterCount:
            gradient = self.calculateGradient()
            m = np.array([atom.coordinates for atom in self.listOfAtoms])
            tangentGrad = np.array([gradient[i] - np.dot(gradient[i], m[i]) * m[i] for i in range(self.size)])
            normOfGrad = list(map(np.linalg.norm, tangentGrad))
            for i in range(self.size):
                if np.dot(self.velocity[i], tangentGrad[i]) < 0:
                    self.velocity[i] = [0, 0, 0]
                else:
                    self.velocity[i] += tangentGrad[i] * delta
                    self.velocity[i] = np.dot(self.velocity[i], tangentGrad[i]) * tangentGrad[i] / (normOfGrad[i] ** 2)
            step = self.velocity * delta + tangentGrad * (delta ** 2)
            n = [np.cross(self.listOfAtoms[i].coordinates, -step[i]) for i in range(self.size)]
            n = list(map(lambda x: x / np.linalg.norm(x), n))
            for j in range (self.size):
                m = self.listOfAtoms[j].coordinates
                angle = np.arctan(np.linalg.norm(step))
                if angle > theta: angle = theta
                self.listOfAtoms[j].coordinates = np.dot(m, rotationMatrix(n[j], angle))
            isConsist = True
            for i in range(self.size):
                isConsist &= normOfGrad[i] < eps
            itr += 1
        self.velocity = np.zeros((self.size, 3))

    def climbing(self):
        print "Starting CI-GNEB"
        itr = 0
        isConsist = False
        while not isConsist and itr < maxIterCount:
            gradient = self.calculateGradient()
            m = np.array([atom.coordinates for atom in self.listOfAtoms])
            tangentGrad = np.array([gradient[i] - np.dot(gradient[i], m[i]) * m[i] for i in range(self.size)])
            force = np.array([tangentGrad[i] - 2 * np.dot(tangentGrad[i], self.tau[i]) * self.tau[i] for i in range(self.size)])
            normOfForce = map(np.linalg.norm, force)
            step = force * delta
            n = [np.cross(self.listOfAtoms[i].coordinates, -step[i]) for i in range(self.size)]
            n = map(lambda x: x / np.linalg.norm(x), n)
            for j in range (self.size):
                m = self.listOfAtoms[j].coordinates
                angle = np.arctan(np.linalg.norm(step))
                if angle > theta: angle = theta
                self.listOfAtoms[j].coordinates = np.dot(m, rotationMatrix(n[j], angle))
            self.energy = self.calculateEnergy()
            self.tau = self.calculateTau()
            isConsist = True
            maxForce = 0
            for i in range(self.size):
                isConsist &= normOfForce[i] < eps
                if maxForce > normOfForce[i]: maxForce = normOfForce[i]
            itr += 1

class Cluster:
    def __init__(self, listOfImages):
        self.size = len(listOfImages)
        self.listOfImages = listOfImages
    
    def addImage(self, image):
        self.size += 1
        self.listOfImages.append(image)
    
    def initNeighbors(self):
        for i in range(1, self.size - 1):
            self.listOfImages[i].previous = self.listOfImages[i - 1]
            self.listOfImages[i].next = self.listOfImages[i + 1]
            self.listOfImages[i].tau = self.listOfImages[i].calculateTau()
    def relaxation(self):
        self.initNeighbors()
        isConsist = False
        itr = 0
        start = time.time()
        while not isConsist and itr < maxIterCount:
            for i in range(1, self.size - 1):
                self.listOfImages[i].calculateForce()
            isConsist = True
            for i in range(1, self.size - 1):
                isConsist &= self.listOfImages[i].makeStep('quick-min')
            print "Iteration: %i" % itr
            itr += 1
            maxForce = 0
            for i in range(1, self.size - 1):
                for j in range(self.listOfImages[0].size):
                    force = np.linalg.norm(self.listOfImages[i].force[j])
                if force > maxForce: maxForce = force
            print "Max force:", maxForce
        print "The MEP was found in %.5f seconds" % (time.time() - start)

    def interpolation(self):
        self.initNeighbors()
        lmbd = [0]
        result = []
        t = []
        for image in self.listOfImages[1:-1]:
            imageCur = np.array([atom.coordinates for atom in image.listOfAtoms])
            imagePrv = np.array([atom.coordinates for atom in image.previous.listOfAtoms])
            lmbd.append(lmbd[-1] + distance(imageCur, imagePrv))
        print "Reaction coordinate interval:", lmbd[1], lmbd[-1]
        for i in range(1, len(lmbd) - 1):
            left = i
            right = i + 1
            imageRight = self.listOfImages[right]
            imageLeft = self.listOfImages[left]
            deltaLeft = np.dot(-imageLeft.calculateGradient().flatten(), imageRight.tau.flatten())
            deltaRight = np.dot(-imageRight.calculateGradient().flatten(), imageLeft.tau.flatten())
            energyLeft = imageLeft.calculateEnergy()
            energyRight = imageRight.calculateEnergy()
            a = -2 * (energyRight - energyLeft) / (lmbd[right] - lmbd[left]) ** 3 + (deltaRight + deltaLeft) / (lmbd[right] - lmbd[left]) ** 2
            b = 3 * (energyRight - energyLeft) / (lmbd[right] - lmbd[left]) ** 2 - (deltaRight + 2 * deltaLeft) / (lmbd[right] - lmbd[left])
            c = deltaLeft
            d = energyLeft
            x = np.linspace(lmbd[left], lmbd[right], 100, endpoint=False)
            t.append(x)
            result.append(a * ((x - lmbd[left]) ** 3) + b * ((x - lmbd[left]) ** 2) + c * (x - lmbd[left]) + d)
        t = np.array(t)
        return np.array(result).flatten(), t.flatten()

def distance(a, b):
    d = 0
    for i in range(len(a)):
        d += ((np.arctan2(np.linalg.norm(np.cross(a[i], b[i])), np.dot(a[i], b[i])) + 2 * np.pi) % (2 * np.pi)) ** 2
    return np.sqrt(d)

def rotationMatrix(vector, angle):
    x = vector[0]
    y = vector[1]
    z = vector[2]
    return np.array([[x ** 2 + np.cos(angle) * (1 - x ** 2), x * y * (1 - np.cos(angle)) - z * np.sin(angle), x * z * (1 - np.cos(angle)) + y * np.sin(angle)], [x * y * (1 - np.cos(angle)) + z * np.sin(angle), y ** 2 + np.cos(angle) * (1 - y ** 2), y * z * (1 - np.cos(angle)) - x * np.sin(angle)], [x * z * (1 - np.cos(angle)) - y * np.sin(angle), y * z * (1 - np.cos(angle)) + x * np.sin(angle), z ** 2 + np.cos(angle) * (1 - z ** 2)]])