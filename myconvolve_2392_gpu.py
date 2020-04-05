import numpy
from pyfftw.interfaces.numpy_fft import fft, ifft
import soundfile
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt



#################
#               #
#  ΕΡΩΤΗΜΑ Α.1  #
#               #
#################

N = 11
while N <= 10:
    N = int(input('Παρακαλώ εισάγετε ένα τυχαίο ακέραιο αριθμό Ν μεγαλύτερο του 10: '))

# αρχικοποίηση των πινάκων εισόδου με τυχαίες τιμές
A = numpy.random.randint(low=-N, high=N, size=N)
B = numpy.random.randint(low=-N, high=N, size=5)


# Υλοποίηση συνέλιξης σύμφωνα με τον ορισμό
def myconvolve_simple(a, b):
    # μεγέθη των συναρτησεων εισόδου
    n = len(a)
    m = len(b)
    # μέγεθος αποτελέσματος της συνέλιξης
    result_size = n + m - 1
    # μηδενισμός/αρχικοποίηση πίνακα εξόδου
    c = numpy.zeros(result_size)
    # προσθήκη επιπλέον μηδενικών στοιχείων στο τέλος των πινάκων εισόδου για την σωστή επανάληψη της for παρακάτω
    a = numpy.pad(a, (0, m - 1), 'constant')
    b = numpy.pad(b, (0, n - 1), 'constant')
    # υλοποίηση συνέλιξης με τον ορισμό(ο δείκτης j αυξάνεται όσο ο δείκτης k μειώνεται σε καθε ένα βήμα αύξησης του
    # δείκτη i)
    for i in range(0, result_size):
        for j, k in zip(range(0, i + 1), range(i, -1, -1)):
            c[i] += a[j] * b[k]
    return c


# Υλοποίηση συνέλιξης με μετασχηματσιμούς Fourier και αντιστροφο μετασχηματισμό Fourier
def myconvolve_fft(a, b):
    # μεγέθη των συναρτησεων εισόδου
    n = len(a)
    m = len(b)
    # μέγεθος αποτελέσματος της συνέλιξης
    result_size = n + m - 1
    # μηδενισμός/αρχικοποίηση πίνακα εξόδου
    c = numpy.zeros(result_size)
    # προσθήκη επιπλέον μηδενικών στοιχείων στο τέλος των πινάκων εισόδου για την σωστή επανάληψη της for παρακάτω
    a = numpy.pad(a, (0, m - 1), 'constant')
    b = numpy.pad(b, (0, n - 1), 'constant')
    fr_a = fft(a)
    fr_b = fft(b)
    cc = ifft(fr_a * fr_b)
    # επανάληψη για νάλάβουμε μόνο τα πραγματικά μέρη των αριθμών
    for i in range(0, len(c)):
        c[i] = numpy.real(cc[i])
    return c


# εκτπύπωση της εξόδου της συναρτησης με τα τυχαία δεδομένα
print('Τυχαίος πίνακας Α:', A)
print('Τυχαίος πίνακας Β:', B)
C1 = myconvolve_simple(A, B)
print('Αποτέλεσμα συνέλιξης με Α και Β με τον ορισμό:', C1)
C2 = myconvolve_fft(A, B)
print('Αποτέλεσμα συνέλιξης με Α και Β με χρήση μετασχηματισμού Fourier:', C2)

##########################
#                        #
#  ΤΕΛΟΣ ΕΡΩΤΗΜΑΤΟΣ Α.1  #
#                        #
##########################

#################
#               #
#  ΕΡΩΤΗΜΑ Α.2  #
#               #
#################

# για χρονομέτρηση
start = cuda.Event()
end = cuda.Event()

# ανάγνωση αρχείων και αποθήκευση των σημάτων και συχνοτητων δειγματοληψιας σε κανονικους πίνακες
sample_audio, saFs = soundfile.read('sample_audio.wav')
pink_noise, pnFs = soundfile.read('pink_noise.wav')

# ο χρόνος εκτέλεσης είναι υπερβολικός για τον υπολογισμό σύμφωνα με τον ορισμό με χρήση μονο της CPU
# start.record()  # χρονομέτρηση cpu myconvolve_simple
# start.synchronize()
# pinkNoise_sampleAudio_simple = myconvolve_simple(sample_audio, pink_noise)
# end.record()  # τέλος χρονομέτρησης cpu myconvolve_simple
# end.synchronize()
#
# secs = start.time_till(end) * 1e-3
# print("myconvolve_simple time :")
# print("%fs \n" % secs)
# print(pinkNoise_sampleAudio_simple[-3:], "\n\n")

start.record()  # χρονομέτρηση cpu myconvolve_fft
start.synchronize()
pinkNoise_sampleAudio_fft = myconvolve_fft(sample_audio, pink_noise)
end.record()  # τέλος χρονομέτρησης cpu myconvolve_fft
end.synchronize()

soundfile.write('pinkNoise_sampleAudio.wav', pinkNoise_sampleAudio_fft, saFs)

secs = start.time_till(end) * 1e-3
print("myconvolve_fft time :")
print("%fs \n" % secs)
print(pinkNoise_sampleAudio_fft[-3:], "\n\n")

# Δημιουργία λευκού θορύβου και συνέλιξη με το σήμα ήχου sample_audio.wav
mean = 0
std = 1
num_whitenoise = 1000
whitenoise = numpy.random.normal(mean, std, size=num_whitenoise)
whiteNoise_sampleAudio_fft = myconvolve_fft(sample_audio, whitenoise)
soundfile.write('whiteNoise_sampleAudio.wav', whiteNoise_sampleAudio_fft, saFs)


###############
#             #
#  ΕΡΩΤΗΜΑ Β  #
#             #
###############

blocks = 2048
block_size = 1024

# Το κομμάτι κώδικα που τρέχει σε κάθε thread της καρτας γραφικων (kernel-κώδικας C++)
mod = SourceModule("""
__global__ void cuda_myconvolve_simple(float *dest, float *a, float *b, int n_iter)
{
  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i<=n_iter){
    for (int j = 0, k = i; j < i + 1 && k > -1; j++, k--) {
      dest[i] += a[j] * b[k];
    }
  }
}
""")

# ορισμός συνάρτησης για να την καλέσω απο python
cuda_myconvolve_simple = mod.get_function("cuda_myconvolve_simple")

# προετοιμασία πινάκων
nSA = len(sample_audio)
mPN = len(pink_noise)
sample_audio = numpy.array(sample_audio, dtype=numpy.float32)
pink_noise = numpy.array(pink_noise, dtype=numpy.float32)
sample_audio = numpy.pad(sample_audio, (0, mPN - 1), 'constant')
pink_noise = numpy.pad(pink_noise, (0, nSA - 1), 'constant')
result_sizeCU = nSA + mPN - 1
# ο πίνακας που θα έχει το αποτέλεσμα της συνέλιξης
pinkNoise_sampleAudio_cuda = numpy.array(numpy.zeros(result_sizeCU), dtype=numpy.float32)

start.record()  # χρονομέτρηση gpu
cuda_myconvolve_simple(cuda.Out(pinkNoise_sampleAudio_cuda), cuda.In(sample_audio), cuda.In(pink_noise),
                       numpy.int32(result_sizeCU), grid=(blocks, 1), block=(block_size, 1, 1))
end.record()  # τέλος χρονομέτρησης gpu
end.synchronize()

soundfile.write('pinkNoise_sampleAudio_cuda.wav', pinkNoise_sampleAudio_cuda, saFs)

secs = start.time_till(end) * 1e-3
print("cuda_myconvolve_simple time :")
print("%fs \n" % secs)
print(pinkNoise_sampleAudio_cuda[-3:], "\n\n")

# Προβολή των γραφημάτων για το ερώτημα Α.1
plt.figure()
plt.subplot(311)
plt.plot(A, 'b')
plt.gca().set_title('Τυχαίος πίνακας Α')
plt.grid(True)

plt.subplot(312)
plt.plot(B, 'g')
plt.gca().set_title('Τυχαίο διάνυσμα Β')
plt.grid(True)

plt.subplot(313)
plt.plot(C1, 'c')
plt.gca().set_title('Συνέλιξη Α και Β')
plt.grid(True)
plt.show()

# Προβολή των γραφημάτων για το ερώτημα Α.2
plt.figure()
plt.subplot(411)
plt.gca().set_title('Σήμα ήχου sample_audio.wav')
plt.plot(sample_audio, 'b')
plt.grid(True)

plt.subplot(412)
plt.gca().set_title('Σήμα ήχου pink_noise.wav')
plt.plot(pink_noise, 'g')
plt.grid(True)

plt.subplot(413)
plt.gca().set_title('Συνέλιξη ήχου sample_audio.wav και του ήχου pink_noise.wav')
plt.plot(pinkNoise_sampleAudio_fft, 'c')
plt.grid(True)
plt.show()

plt.subplot(414)
plt.gca().set_title('Συνέλιξη ήχου sample_audio.wav και του ήχου white_noise.wav')
plt.plot(whiteNoise_sampleAudio_fft, 'c')
plt.grid(True)
plt.show()

# Προβολή των γραφημάτων για το ερώτημα Α.2 με την συνάρτηση cuda_Myconvolve_simple
plt.figure()
plt.subplot(311)
plt.gca().set_title('(cuda) Σήμα ήχου sample_audio.wav')
plt.plot(sample_audio, 'b')
plt.grid(True)

plt.subplot(312)
plt.gca().set_title('(cuda) Σήμα ήχου pink_noise.wav')
plt.plot(pink_noise, 'g')
plt.grid(True)

plt.subplot(313)
plt.gca().set_title('(cuda) Συνέλιξη ήχου sample_audio.wav και του ήχου white_noise.wav')
plt.plot(pinkNoise_sampleAudio_cuda, 'c')
plt.grid(True)
plt.show()
