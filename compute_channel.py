from numpy import log10, sqrt, pi, floor
import numpy as np
from math import sqrt, cos, sin, pi, log10
from cmath import exp
from scipy.constants import speed_of_light
from numpy.random import shuffle, uniform, choice, normal, rand
from typing import Union
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import mpld3


def calculate_3gpp_antenna(aoa_az, aoa_el, N, wavelength):
    """This function creates a directional antenna pattern depending on the
    number of antenna elements specified. The total gain is calculated as the
    superposition of elements. Therefore, the beam becomes more narrow with
    larger number given"""
    d_h = wavelength/2
    d_v = d_h
    N_hor = N
    N_vert = N
    Am = 30     # the front-back ratio
    SLA = 30    # the lower limit
    G = 0       # maximum gain of an antenna element
    ang_3db = 65*pi/180
    A_eh = -min(12 * ((aoa_az / ang_3db) ** 2), Am)
    A_ev = -min(12 * ((aoa_el / ang_3db) ** 2), SLA)
    # Compute a magnitude of an element pattern
    P_E = G - min(-(A_eh + A_ev), Am)
    P_e = 10**(P_E/20)

    a = []
    # Calculate phase shift and weighting factor for all array elements
    for m in range(0,N_hor):
        for n in range(0, N_vert):
            v = exp(-2j*pi*((n-1)*cos(aoa_el)*(d_v/wavelength) + (m-1)*sin(aoa_el)*sin(aoa_az)*(d_h/wavelength) ))
            w = (1/sqrt(N_hor*N_vert))*exp(2j*pi*((n-1)*cos(aoa_el)*(d_v/wavelength) +
                                                  (m-1)*cos(aoa_el)*sin(aoa_az)*(d_h/wavelength)))
            E_mn = P_e*v*w
            a.append(E_mn)

    # Calculate the gain as the superposition of the elements
    G = 20*log10(abs(sum(a)))
    return G


def norm(v: np.ndarray) -> [np.ndarray, float]:
    """Normalises the value"""
    return np.sqrt((v * v).sum(axis=0))


def vector_normalize(v: np.ndarray) -> np.ndarray:
    """Normalises the vector"""
    n = norm(v)
    assert n > 0, "Can not normalize null vector!"
    return v / norm(v)


def DB2RATIO(d):
    """Performs conversion from dB to linear scale for better visual
    interpretation of the code"""
    return 10.0 ** (d / 10.0)


def RATIO2DB(x):
    """Performs conversion linear scale to dB for better visual
    interpretation of the code"""
    return 10.0 * np.log10(x)


def cart2sph(x, y, z):
    """Converts Cartesian coordinates into spherical"""
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(az, el, r=1):
    """Converts spherical coordinates into Cartesian"""
    x = r*sin(el)*cos(az)
    y = r*sin(el)*sin(az)
    z = r*cos(el)
    return x, y, z


def friis_path_loss_dB(dist: [float, np.ndarray], frequency_Hz: float, n: float = 2.0) -> Union[float, np.ndarray]:
    """Friis path loss formula for calibration"""
    return RATIO2DB(np.power(speed_of_light / (frequency_Hz * 4 * pi * dist), n))


class MP_chan_params(object):
    """This class defines basic parameters of the channel"""
    def __init__(self, carrier_frequency_Hz, N_clust, N_rays):
        self.N_clust = N_clust  # number of clusters
        self.N_rays = N_rays    # number of rays per cluster

        self.prop_loss_function = friis_path_loss_dB
        self.coverage_n = 2.0   # Path loss exponent for LOS
        self.delay_scaling = max(0.25, 6.5622-3.4084*log10(carrier_frequency_Hz))  # delay scaling parameter 3
        self.ds_var = 0.66      # delay spread (mean)
        self.ds_mean = -6.955 - 0.0963*log10(carrier_frequency_Hz)
        self.per_clust_sh = 3   # sf parameter
        self.ricean_fact_mean = 9  # ricean K factor
        self.ricean_fact_var = 3.5
        self.max_gain = 100.
        self.max_SINR = 40      # dB
        self.min_SINR = -30     # dB

        self.xpr_mean = 8       # cross polarization ration (mean)
        self.xpr_var = 4.       # variance

        # angular spread in azimuth and zenith planes
        self.asd_mean = 1.06 + 0.114*log10(carrier_frequency_Hz)
        self.asd_var = 0.28
        self.asa_mean = 1.81
        self.asa_var = 0.20

        self.carrier = carrier_frequency_Hz
        self.wavelength = speed_of_light/self.carrier


class MP_Chan_State(object):
    """This class stores the output of the propagation model"""
    def __init__(self):
        self.PL = 0
        self.phase_delay = 0


def generate_clusters(self, carrier_frequency_Hz: float, d, dv, PL_type: str):
    """This function generates clusters according to the 3GPP TR 38.901 and returning delay, power values, ect."""
    res = MP_Chan_State()
    # First, larger-scale parameters are defined
    los_ray_pow = 0
    asd_spread = 1.06 + 0.114*log10(carrier_frequency_Hz)
    asa_spread = 1.81
    zsd_spread = 0

    zsa_spread = 0.95
    angle_offset = [0.0447, -0.0447,
                    0.1413, -0.1413,
                    0.2492, -0.2492,
                    0.3715, -0.3715,
                    0.5129, -0.5129,
                    0.6797, -0.6797,
                    0.8844, -0.8844,
                    1.1481, -1.1481,
                    1.5195, -1.5195,
                    2.1551, -2.1551]

    zsd_mean = 0
    zsd_var = 0.32

    zsa_mean = 0.95
    zsa_var = 0.16

    zbd_mean = 0
    zbd_var = 0.3

    zba_mean = 0
    zba_var = 0.3

    normal_var_vect = [normal(loc=0., scale=1.) for _ in range(0, 7)]
    # Correlation matrix provides consistency to this model
    cor_matrix = [[1.0, 0.49, 0, 0.29, 0, 0.2, -0.26],
                  [0.49, 1.0, 0, 0.62, 0, 0.47, -0.16],
                  [0, 0, 1.0, 0, 0.55, 0, 0.45],
                  [0.29, 0.62, 0, 1.0, 0.33, 0.76, 0.17],
                  [0, 0, 0.55, 0.33, 1.0, 0.33, 0.69],
                  [0.2, 0.47, 0, 0.76, 0.33, 1.0, 0.39],
                  [-0.26, -0.16, 0.45, 0.17, 0.69, 0.39, 1.0]]
    V, D = np.linalg.eigh(cor_matrix, UPLO='U')
    cor_multiplier = np.dot(D, np.sqrt(np.diag(V)))
    normal_var_vect = np.dot(cor_multiplier, normal_var_vect)
    PL = self.get_LOS_channel(d).PL

    d_spread = 10 ** ((self.params.ds_var * normal_var_vect[0]) + self.params.ds_mean)

    asd = min(10 ** ((self.params.asd_var * normal_var_vect[1]) + self.params.asd_mean), 100.0)
    asa = min(10 ** ((self.params.asa_var * normal_var_vect[2]) + self.params.asa_mean), 100.0)
    zsd = min(10 ** ((zsd_var * normal_var_vect[3]) + zsd_mean), 40.0)
    zsa = min(10 ** ((zsa_var * normal_var_vect[4]) + zsa_mean), 40.0)
    m_bias_zod = max(-10 ** (zbd_mean + zbd_var * normal_var_vect[5]), -75)
    m_bias_zoa = max(-10 ** (zba_mean + zba_var * normal_var_vect[6]), -75)
    ricean_fact = abs((self.params.ricean_fact_var * normal_var_vect[5]) + self.params.ricean_fact_mean)

    # Then, ray-related parameters are generated
    outgoing_ray = dv
    outgoing_ray = vector_normalize(outgoing_ray)
    los_aod, los_zod, r = cart2sph(outgoing_ray[0], outgoing_ray[1],
                                   outgoing_ray[2])
    los_aod = np.degrees(los_aod)
    los_zod = np.degrees(los_zod)

    incoming_ray = dv
    incoming_ray = vector_normalize(incoming_ray)
    los_aoa, los_zoa, r = cart2sph(-incoming_ray[0], -incoming_ray[1],
                                   incoming_ray[2])
    los_aoa = np.degrees(los_aoa)
    los_zoa = np.degrees(los_zoa)

    cluster_delays, cluster_powers_init = [], []
    cluster_aoas, cluster_aods = [], []
    cluster_zoas, cluster_zods = [], []

    cluster_delays_list = []
    cluster_aoas_list, cluster_aods_list = [], []
    cluster_zoas_list, cluster_zods_list = [], []
    cluster_xpr_list = []
    cluster_phiVV_list, cluster_phiVH_list = [], []
    cluster_phiHV_list, cluster_phiHH_list = [], []

    for i in range(0, self.params.N_clust):
        delay_var = uniform(0.000000001, 1.0)
        delay = -self.params.delay_scaling * d_spread * np.log(delay_var)
        cluster_delays.append(delay)
    # The MPCs are stored according to the arrival time
    cluster_delays.sort()
    if PL_type == 'NLOS':
        res.phase_delay = np.array(cluster_delays).min()
        cluster_delays -= res.phase_delay

    for i in range(0, len(cluster_delays)):
        cluster_delays_new = []
        for j in range(0, self.params.N_rays):
            delay = cluster_delays[i]
            if 4 < j <= 7:
                delay += 5e-9
            elif j > 7:
                delay += 10e-9
            cluster_delays_new.append(delay)
        cluster_delays_list.append(cluster_delays_new)

    for i in range(0, self.params.N_clust):
        # The power values are computed based on the computed delay and the defined delay spread and scaling, i.e.,
        # the MPC, which arrives earlier is typically associated with higher power
        z_clust = normal(0, self.params.per_clust_sh ** 2)
        power = (10 ** (-0.1 * z_clust)) * np.exp(
            -cluster_delays[i] * (self.params.delay_scaling - 1) / (self.params.delay_scaling * d_spread))
        cluster_powers_init.append(power)

    cluster_powers = list(np.array(cluster_powers_init) / sum(cluster_powers_init))

    for i in range(0, self.params.N_clust):
        # This part computes departure and arrival angles (AoDs and AoAs)
        xn = [-1, 1]
        uni_rand_int = choice(xn)
        norm_rand_var = normal(0, asd / 7)
        aod = 2.33 * (asd / 1.4) * np.sqrt(-np.log(cluster_powers[i] / max(cluster_powers)))
        aod = (uni_rand_int * aod) + norm_rand_var + los_aod

        uni_rand_int = choice(xn)
        norm_rand_var = normal(0, asa / 7)
        aoa = 2.33 * (asa / 1.4) * np.sqrt(-np.log(cluster_powers[i] / max(cluster_powers)))
        aoa = (uni_rand_int * aoa) + norm_rand_var + los_aoa

        uni_rand_int = choice(xn)
        norm_rand_var = normal(0, zsd / 7)
        zod = -1.01 * zsd * np.log(cluster_powers[i] / max(cluster_powers))
        zod = (uni_rand_int * zod) + norm_rand_var + los_zod + m_bias_zod

        uni_rand_int = choice(xn)
        norm_rand_var = normal(0, zsa / 7)
        zoa = -1.01 * zsa * np.log(cluster_powers[i] / max(cluster_powers))
        zoa = (uni_rand_int * zoa) + norm_rand_var + los_zoa + m_bias_zoa

        cluster_aoas.append(aoa)
        cluster_aods.append(aod)
        cluster_zoas.append(zoa)
        cluster_zods.append(zod)

        aoa_list, aod_list = [], []
        zoa_list, zod_list = [], []
        xpr_list = []
        phiVV_list, phiVH_list = [], []
        phiHV_list, phiHH_list = [], []

        for j in range(0, self.params.N_rays):
            # After the angles are generated, the offset is added per each ray within the cluster
            xn = [1, -1]
            aod_ray = aod + asd_spread * xn[j % len(xn)] * angle_offset[int(floor(j / 2))]
            aoa_ray = aoa + asa_spread * xn[j % len(xn)] * angle_offset[int(floor(j / 2))]
            zoa_ray = zoa + zsa_spread * xn[j % len(xn)] * choice(xn) * angle_offset[int(floor(j / 2))]
            zod_ray = zod + zsd_spread * xn[j % len(xn)] * choice(xn) * angle_offset[int(floor(j / 2))]
            aoa_list.append(aoa_ray)
            aod_list.append(aod_ray)
            zoa_list.append(zoa_ray)
            zod_list.append(zod_ray)
            xpr = self.params.xpr_var * normal(0., 1.) + self.params.xpr_mean
            xpr = 10 ** (xpr / 10)
            xpr_list.append(xpr)
            phiVV = -pi + 2 * pi * rand()
            phiVH = -pi + 2 * pi * rand()
            phiHV = -pi + 2 * pi * rand()
            phiHH = phiVV + (rand() > 0.5) * pi
            phiVV_list.append(phiVV)
            phiVH_list.append(phiVH)
            phiHV_list.append(phiHV)
            phiHH_list.append(phiHH)

        shuffle(aoa_list)
        shuffle(aod_list)
        shuffle(zoa_list)
        shuffle(zod_list)
        cluster_aoas_list.append(aoa_list)
        cluster_aods_list.append(aod_list)
        cluster_zoas_list.append(zoa_list)
        cluster_zods_list.append(zod_list)
        cluster_xpr_list.append(xpr_list)
        cluster_phiVV_list.append(phiVV_list)
        cluster_phiVH_list.append(phiVH_list)
        cluster_phiHV_list.append(phiHV_list)
        cluster_phiHH_list.append(phiHH_list)

    if PL_type == 'LOS':
        los_ray_pow = ricean_fact / (1 + ricean_fact)
        cluster_powers.insert(0, los_ray_pow)
        cluster_aoas.insert(0, los_aoa)
        cluster_aods.insert(0, los_aod)
        cluster_zoas.insert(0, los_zoa)
        cluster_zods.insert(0, los_zod)

        cluster_delays_list.insert(0, [0])
        cluster_aoas_list.insert(0, [los_aoa])
        cluster_aods_list.insert(0, [los_aod])
        cluster_zoas_list.insert(0, [los_zoa])
        cluster_zods_list.insert(0, [los_zod])
        phiVV = -pi + 2 * pi * rand()
        phiVH = 0
        phiHV = 0
        phiHH = phiVV + pi
        cluster_xpr_list.insert(0, [float('inf')])
        cluster_phiVV_list.insert(0, [phiVV])
        cluster_phiVH_list.insert(0, [phiVH])
        cluster_phiHV_list.insert(0, [phiHV])
        cluster_phiHH_list.insert(0, [phiHH])

    N_clusters = self.params.N_clust
    cluster_aods = np.array(cluster_aods_list)
    cluster_aoas = np.array(cluster_aoas_list)
    cluster_zods = np.array(cluster_zods_list)
    cluster_zoas = np.array(cluster_zoas_list)

    cluster_delays_list = np.array(cluster_delays_list)
    cluster_xpr_list = np.array(cluster_xpr_list)
    cluster_phiVV_list = np.array(cluster_phiVV_list)
    cluster_phiVH_list = np.array(cluster_phiVH_list)
    cluster_phiHV_list = np.array(cluster_phiHV_list)
    cluster_phiHH_list = np.array(cluster_phiHH_list)

    # Gather all the parameters
    result = [res, PL, PL_type, N_clusters, cluster_powers, cluster_aoas, cluster_aods, cluster_zoas,
              cluster_zods, cluster_phiVV_list, cluster_phiVH_list, cluster_phiHV_list, cluster_phiHH_list,
              cluster_xpr_list, cluster_delays_list, los_ray_pow]
    return result


class MP_Propagation_Model(object):
    """This class is used to calculate the final channel state. It employs the functions
     and classes defined above. It performs weighting of the MPC components according to an antenna pattern,
     ads a LoS component and  performs power correction
    """

    def __init__(self, params: MP_chan_params, Nant_tx, Nant_rx):
        # This defines the channel parameters and the number of antenna elements at the Tx and Rx
        self.params = params
        self.Nant_tx = Nant_tx
        self.Nant_rx = Nant_rx

    def get_LOS_channel(self, dist):
        # Computes the LoS component
        PL = -self.params.prop_loss_function(dist, self.params.carrier, self.params.coverage_n)
        cs = MP_Chan_State()
        cs.PL = PL
        cs.phase_delay = dist / speed_of_light
        return cs

    def get_Cluster_channel_mmWave(self, carrier_frequency_Hz, src_pos, dst_pos, PL_type):
        # Weighting of the MPC components, power correction, and LoS path are computed within this method
        dv = src_pos - dst_pos
        dist = np.linalg.norm(dv)
        direction = dv/dist     # It is assumed that Tx and Rx are aligned, which allows computing
                                # attenuation of other MPCs arriving at the antenna

        # The direction is converted into the angular coordinates and provided as an input to an antenna model
        azimuth_angle_tx, zenith_angle_tx, r = cart2sph(direction[0], direction[1], direction[2])
        tx_ant = [azimuth_angle_tx, zenith_angle_tx]

        azimuth_angle_rx, zenith_angle_rx, r = cart2sph(-direction[0], -direction[1], direction[2])
        rx_ant = [azimuth_angle_rx, zenith_angle_rx]

        clusters = generate_clusters(self, carrier_frequency_Hz, dist, dv, PL_type)

        res, PL, PL_type, N_clusters, cluster_powers, cluster_aoas, cluster_aods, cluster_zoas, cluster_zods, \
        cluster_phiVV_list, cluster_phiVH_list, cluster_phiHV_list, cluster_phiHH_list, \
        cluster_xpr_list, cluster_delays_list, los_ray_pow = clusters

        ir = []
        cluster_tx_gains_list, cluster_rx_gains_list = [], []
        for i in range(0, len(cluster_aods)):
            # Weighting of MPCs according to their angles and direction
            # of antenna patterns are computed for the Tx and Rx
            tx_gains_list = []
            rx_gains_list = []
            for j in range(0, len(cluster_aods[i])):
                tx_gains = calculate_3gpp_antenna(tx_ant[0] - (cluster_aods[i][j] * pi / 180),
                                                  tx_ant[1] - (cluster_zods[i][j] * pi / 180),
                                                  self.Nant_tx, self.params.wavelength)
                tx_gains = DB2RATIO(tx_gains)
                rx_gains = calculate_3gpp_antenna(rx_ant[0] - (cluster_aoas[i][j] * pi / 180),
                                                  rx_ant[1] - (cluster_zoas[i][j] * pi / 180),
                                                  self.Nant_rx, self.params.wavelength)
                rx_gains = DB2RATIO(rx_gains)

                tx_gains_list.append(tx_gains)
                rx_gains_list.append(rx_gains)
            cluster_tx_gains_list.append(tx_gains_list)
            cluster_rx_gains_list.append(rx_gains_list)
        cluster_rx_gains_list = np.array(cluster_rx_gains_list)
        cluster_tx_gains_list = np.array(cluster_tx_gains_list)

        for i in range(0, len(cluster_tx_gains_list)):
            ray_values = []
            for j in range(0, len(cluster_tx_gains_list[i])):
                if PL_type == 'LOS' and i == 0:
                    pol_matrix = np.array([[np.exp(1j * cluster_phiVV_list[i][j]), 0],
                                           [0, -1 * np.exp(1j * cluster_phiHH_list[i][j])]])
                else:
                    pol_matrix = np.array([[np.exp(1j * cluster_phiVV_list[i][j]),
                                            sqrt(cluster_xpr_list[i][j] ** -1) * np.exp(
                                                1j * cluster_phiVH_list[i][j])],
                                           [sqrt(cluster_xpr_list[i][j] ** -1) * np.exp(
                                               1j * cluster_phiHV_list[i][j]),
                                            np.exp(1j * cluster_phiHH_list[i][j])]])

                tx_gains = cluster_tx_gains_list[i][j]
                rx_gains = cluster_rx_gains_list[i][j]
                f_tx = np.array([1, 0]).T
                f_rx = np.array([1, 0])

                assert tx_gains >= 0 and rx_gains >= 0
                test1 = np.dot(np.dot(f_rx, pol_matrix), f_tx)
                result = (test1 * tx_gains * rx_gains) ** 0.5
                ray_values.append(result)
            los_mult = 1
            if PL_type == 'LOS':
                los_mult = sqrt(1 / (1 + los_ray_pow))
            imp_r_val = sum(np.abs(
                los_mult * sqrt(cluster_powers[i] / len(cluster_tx_gains_list[i])) * np.array(ray_values)) ** 2)
            ir.append(imp_r_val)
        ir = np.array(ir)
        IR_power_correction = RATIO2DB((np.sum(ir)))

        res.MPC_delays = [el + dist/speed_of_light for el in cluster_delays_list]
        res.MPC_powers = np.array(cluster_powers)
        res.PL = PL - IR_power_correction
        res.Power_correction = IR_power_correction
        res.MPC_aoas = np.array(cluster_aoas)
        res.MPC_zoas = np.array(cluster_zoas)
        res.tx_rx_dist = dist
        if PL_type == 'LOS':
            res.PL_type = 1
        else:
            res.PL_type = 0
        return res


def run_channel(f_carrier, dst_pos, src_pos, pl_type, N_clusters, N_rays):
    """This functions runs the cluster generation procedure and MPC weighting.
    The final results are displayed in the form of """
    dst_pos = np.array(dst_pos)
    src_pos = np.array(src_pos)
    ch_params = MP_chan_params(f_carrier, N_clusters, N_rays)
    ch_prop_model = MP_Propagation_Model(ch_params, Nant_tx=4, Nant_rx=4)
    Ch_params_clusters = ch_prop_model.get_Cluster_channel_mmWave(f_carrier, dst_pos, src_pos, pl_type)

    fig = go.Figure()
    # Plot destination and source coordinates
    fig.add_trace(go.Scatter3d(x=[dst_pos[0]], y=[dst_pos[1]], z=[dst_pos[2]], mode='markers', name='Rx'))
    fig.add_trace(go.Scatter3d(x=[src_pos[0]], y=[src_pos[1]], z=[src_pos[2]], mode='markers', name='Tx'))
    layout = go.Layout(yaxis=dict(range=[0, 0.4]))
    for n_cluster in range(0, len(Ch_params_clusters.MPC_aoas)):
        d = Ch_params_clusters.tx_rx_dist/2 + 100*np.random.randn()
        # The colors for the 3D plot are generated randomly
        color = tuple(np.random.randint(0, 255, 3))
        c_coord = np.zeros([len(Ch_params_clusters.MPC_aoas[n_cluster]), 3])
        for n_ray in range(0, len(Ch_params_clusters.MPC_aoas[n_cluster])):
            x,y,z=sph2cart(Ch_params_clusters.MPC_aoas[n_cluster][n_ray],
                           Ch_params_clusters.MPC_zoas[n_cluster][n_ray], d)
            c_coord[n_ray, 0] = x
            c_coord[n_ray, 1] = y
            c_coord[n_ray, 2] = z
            # Plot rays from/to each cluster
            fig.add_trace(go.Scatter3d(x=[dst_pos[0], x, src_pos[0]],
                                       y=[dst_pos[1], y, src_pos[1]], z=[dst_pos[2], z, src_pos[2]],
                          marker=dict(size=0.1,
                                      color='rgba' + str(color)[0:-1] + ', 0.3)',
                                      colorscale='Viridis',
                                      opacity=0.8), name='Ray ' + str(n_ray + 1)))
        # Plot clusters
        fig.add_trace(go.Scatter3d(x=[np.mean(c_coord[:,0])], y=[np.mean(c_coord[:,1])], z=[np.mean(c_coord[:,2])],
                      marker=dict(size=20,
                      color='rgba' + str(color)[0:-1] + ', 0.3)',
                      colorscale='Viridis',
                      opacity=0.5), name='Cluster ' + str(n_cluster + 1)))

    fig.show()
    fig.write_html("test.html")

    fig0, ax0 = plt.subplots()
    max_power = max(Ch_params_clusters.MPC_powers)
    x, y = np.array([]), np.array([])
    for i, sample in enumerate(Ch_params_clusters.MPC_delays):
        pn = np.ones(len(sample))*(Ch_params_clusters.MPC_powers[i]*(1/(len(sample))))/max_power
        x = np.append(x, np.array(sample)*1e6)
        y = np.append(y, pn)
    ax0.stem(x, y)
    ax0.set_ylabel('Normalized CIR')
    ax0.set_xlabel('Delay, µs')
    ax0.set_ylim([0,1])
    ax0.grid()
    fig0 = mpld3.fig_to_html(fig0)
    file = open("figure.html","w")
    file.write(fig0)
    file.close()


if __name__ == '__main__':
    f_carrier = 30e9
    dst_pos = np.array([10, 20, 1.5])
    src_pos = np.array([500, 50, 10])
    pl_type = 'LOS'
    N_rays = 5
    N_clusters = 5
    run_channel(f_carrier, dst_pos, src_pos, pl_type, N_clusters, N_rays)
