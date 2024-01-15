import json
import os
import multiprocessing
import itertools
import math
import random

from infinibatch import iterators
from functools import partial
from .lm_loader_v3 import LMLoader
from .utils import NativeCheckpointableIterator, WeightNoRandomStateIterator, EOL_SYMBOL, SelectManyNoSkipIterator
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from tiktoken.core import Encoding

MAX_SHARD_SIZE = 10000000 # 10M files

NON_JSON_SET = ['c4/shard', 'cc-100/shard', 'wiki/shard_v1']
TIKTOKEN_LINE_BREAK = set([198, 271, 280, 317, 319, 340, 341, 345, 382, 397, 401, 457, 464, 512, 517, 534, 545, 557, 627, 629, 633, 696, 702, 720, 740, 741, 746, 756, 761, 803, 881, 886, 891, 933, 947, 987, 997, 1021, 1025, 1035, 1038, 1084, 1125, 1158, 1173, 1177, 1192, 1235, 1240, 1270, 1282, 1287, 1329, 1350, 1363, 1432, 1449, 1454, 1459, 1465, 1473, 1504, 1583, 1602, 1613, 1657, 1679, 1700, 1720, 1763, 1784, 1809, 1827, 1875, 1909, 1967, 1980, 2055, 2195, 2266, 2268, 2313, 2330, 2341, 2355, 2368, 2412, 2451, 2455, 2456, 2499, 2591, 2595, 2608, 2616, 2622, 2634, 2662, 2670, 2820, 2861, 2885, 2892, 2904, 2907, 2950, 3033, 3086, 3108, 3120, 3147, 3155, 3227, 3270, 3291, 3304, 3317, 3356, 3364, 3379, 3382, 3456, 3490, 3540, 3559, 3569, 3591, 3602, 3618, 3638, 3652, 3677, 3694, 3707, 3718, 3730, 3743, 3818, 3840, 3945, 3955, 3961, 4000, 4019, 4077, 4098, 4125, 4260, 4286, 4352, 4390, 4479, 4485, 4489, 4524, 4532, 4555, 4561, 4574, 4649, 4660, 4713, 4743, 4764, 4815, 4828, 4926, 4965, 4982, 4999, 5051, 5062, 5139, 5235, 5240, 5243, 5244, 5322, 5344, 5380, 5537, 5551, 5555, 5572, 5585, 5667, 5680, 5731, 5736, 5779, 5903, 5904, 5984, 5996, 6018, 6032, 6053, 6054, 6087, 6088, 6094, 6121, 6188, 6235, 6242, 6333, 6336, 6343, 6360, 6394, 6408, 6454, 6466, 6470, 6494, 6552, 6557, 6611, 6694, 6698, 6876, 6905, 6939, 7014, 7058, 7071, 7110, 7132, 7171, 7233, 7260, 7275, 7291, 7312, 7361, 7377, 7466, 7468, 7470, 7481, 7511, 7519, 7544, 7663, 7686, 7700, 7722, 7768, 7775, 7786, 7849, 7879, 7887, 7961, 8044, 8181, 8256, 8257, 8295, 8408, 8488, 8555, 8731, 8756, 8786, 8851, 8981, 9000, 9122, 9133, 9174, 9175, 9226, 9432, 9456, 9505, 9522, 9586, 9658, 9723, 9733, 9758, 9763, 9772, 9801, 9837, 9938, 9946, 10080, 10113, 10162, 10197, 10246, 10294, 10342, 10359, 10381, 10450, 10556, 10560, 10586, 10605, 10636, 10661, 10665, 10669, 10792, 10818, 10912, 11111, 11147, 11187, 11192, 11200, 11290, 11367, 11390, 11414, 11444, 11498, 11690, 11740, 11818, 11910, 11950, 12038, 12064, 12068, 12240, 12241, 12275, 12291, 12355, 12403, 12417, 12419, 12423, 12515, 12559, 12578, 12586, 12647, 12662, 12679, 12713, 12795, 12803, 12858, 12872, 12888, 12957, 12996, 13050, 13090, 13188, 13220, 13251, 13300, 13352, 13355, 13407, 13417, 13436, 13503, 13507, 13549, 13801, 14012, 14062, 14182, 14211, 14237, 14342, 14421, 14440, 14501, 14557, 14623, 14665, 14670, 14719, 14781, 14790, 14838, 14852, 14915, 14941, 14963, 15018, 15029, 15044, 15053, 15073, 15092, 15152, 15162, 15276, 15364, 15397, 15399, 15425, 15428, 15434, 15497, 15656, 15658, 15735, 15759, 15776, 15786, 15799, 15804, 15816, 15850, 15882, 15908, 15993, 16049, 16052, 16123, 16129, 16143, 16176, 16240, 16244, 16291, 16300, 16336, 16462, 16484, 16508, 16554, 16575, 16616, 16638, 16764, 16823, 16853, 16918, 16925, 17122, 17288, 17311, 17338, 17350, 17359, 17398, 17473, 17492, 17494, 17556, 17642, 17702, 17707, 17856, 17875, 17934, 17935, 18013, 18020, 18026, 18072, 18086, 18108, 18171, 18218, 18304, 18309, 18325, 18376, 18473, 18490, 18722, 18737, 18760, 18769, 18796, 18840, 18869, 18888, 18903, 18928, 18966, 19014, 19052, 19053, 19066, 19086, 19103, 19124, 19154, 19276, 19279, 19296, 19299, 19327, 19341, 19450, 19485, 19548, 19652, 19681, 19691, 19789, 19800, 19864, 19884, 19888, 19908, 19938, 19985, 20083, 20084, 20106, 20116, 20159, 20168, 20254, 20265, 20386, 20490, 20495, 20679, 20838, 20845, 20923, 20959, 20977, 20979, 20999, 21011, 21274, 21366, 21410, 21421, 21500, 21531, 21537, 21671, 21681, 21711, 21732, 21762, 21812, 21863, 21898, 21932, 22121, 22174, 22242, 22307, 22341, 22390, 22406, 22411, 22414, 22428, 22431, 22438, 22445, 22549, 22623, 22669, 22742, 22825, 22846, 22849, 22859, 22860, 22896, 22953, 23094, 23113, 23123, 23138, 23145, 23169, 23341, 23431, 23438, 23494, 23535, 23547, 23584, 23631, 23704, 23792, 23811, 23843, 23849, 23954, 24010, 24022, 24049, 24226, 24262, 24287, 24314, 24324, 24333, 24356, 24371, 24377, 24413, 24482, 24484, 24518, 24546, 24558, 24688, 24859, 24982, 24984, 25064, 25106, 25158, 25174, 25250, 25293, 25332, 25348, 25374, 25376, 25393, 25441, 25536, 25638, 25758, 25765, 25782, 25833, 25854, 25863, 25881, 25895, 25924, 26027, 26034, 26101, 26145, 26196, 26221, 26315, 26356, 26383, 26389, 26510, 26525, 26543, 26546, 26547, 26570, 26582, 26637, 26652, 26706, 26722, 26727, 26815, 26906, 26914, 26972, 26977, 26986, 26999, 27001, 27064, 27118, 27135, 27164, 27189, 27218, 27333, 27355, 27376, 27381, 27482, 27507, 27567, 27585, 27644, 27676, 27677, 27788, 27829, 27880, 27892, 27907, 27926, 27938, 27948, 27959, 28072, 28105, 28140, 28184, 28212, 28225, 28243, 28283, 28288, 28319, 28348, 28370, 28374, 28411, 28416, 28452, 28465, 28524, 28527, 28586, 28684, 28714, 28768, 28801, 28866, 28871, 28918, 28966, 29001, 29138, 29175, 29194, 29216, 29275, 29307, 29347, 29361, 29411, 29436, 29448, 29472, 29475, 29489, 29494, 29529, 29547, 29620, 29670, 29681, 29765, 29769, 29773, 29812, 30058, 30074, 30078, 30143, 30184, 30210, 30222, 30284, 30340, 30424, 30629, 30655, 30662, 30736, 30916, 30936, 31118, 31134, 31254, 31318, 31347, 31423, 31490, 31538, 31558, 31563, 31593, 31658, 31725, 31734, 31745, 31795, 31836, 31849, 31879, 31893, 31931, 32049, 32091, 32318, 32325, 32339, 32357, 32395, 32396, 32407, 32518, 32583, 32587, 32807, 32815, 32829, 32864, 32897, 33006, 33031, 33080, 33157, 33262, 33303, 33379, 33414, 33486, 33523, 33645, 33648, 33698, 33709, 33716, 33723, 33736, 33905, 33968, 33972, 34113, 34121, 34184, 34193, 34257, 34279, 34299, 34451, 34518, 34536, 34598, 34721, 34726, 34741, 34766, 34773, 34794, 34834, 34900, 34962, 35016, 35033, 35047, 35049, 35126, 35183, 35235, 35241, 35249, 35284, 35400, 35427, 35432, 35441, 35503, 35583, 35599, 35683, 35742, 35749, 35864, 35873, 35921, 35929, 35964, 36085, 36175, 36199, 36215, 36217, 36284, 36308, 36319, 36348, 36397, 36411, 36552, 36628, 36736, 36796, 36818, 36821, 36886, 36929, 36933, 36955, 37280, 37307, 37360, 37423, 37460, 37468, 37574, 37602, 37677, 37722, 37724, 37813, 37881, 37918, 37925, 37945, 37982, 38014, 38028, 38079, 38084, 38138, 38304, 38335, 38354, 38365, 38375, 38465, 38489, 38503, 38545, 38721, 38763, 38792, 38890, 38959, 39060, 39094, 39136, 39185, 39188, 39278, 39325, 39420, 39430, 39486, 39508, 39545, 39597, 39707, 39709, 39876, 39912, 39937, 40042, 40081, 40124, 40171, 40306, 40376, 40390, 40465, 40567, 40614, 40628, 40667, 40669, 40673, 40748, 40763, 40778, 40780, 40824, 40849, 40874, 40881, 40965, 40966, 40987, 41045, 41052, 41074, 41107, 41354, 41404, 41417, 41430, 41437, 41742, 41753, 41822, 41825, 41940, 42001, 42049, 42064, 42125, 42137, 42175, 42229, 42265, 42291, 42296, 42333, 42389, 42444, 42501, 42505, 42546, 42553, 42678, 42720, 42736, 42755, 42793, 42849, 42943, 42953, 43009, 43020, 43040, 43052, 43058, 43112, 43113, 43115, 43171, 43180, 43209, 43232, 43279, 43297, 43352, 43373, 43463, 43484, 43498, 43550, 43619, 43674, 43725, 43933, 43971, 44105, 44157, 44160, 44253, 44296, 44370, 44384, 44441, 44455, 44473, 44497, 44588, 44601, 44708, 44791, 44838, 44986, 45026, 45198, 45199, 45222, 45232, 45242, 45294, 45416, 45460, 45464, 45516, 45711, 45751, 45765, 45802, 45820, 45832, 45835, 45849, 46055, 46093, 46116, 46127, 46200, 46228, 46319, 46420, 46435, 46449, 46485, 46526, 46532, 46534, 46675, 46711, 46726, 46749, 46805, 46822, 46848, 46906, 46907, 46914, 46930, 47061, 47082, 47191, 47251, 47446, 47526, 47542, 47552, 47656, 47744, 47749, 47839, 47896, 47973, 48122, 48165, 48201, 48244, 48284, 48436, 48469, 48523, 48546, 48549, 48556, 48586, 48703, 48716, 48826, 48861, 48957, 49015, 49089, 49164, 49185, 49209, 49215, 49216, 49274, 49342, 49371, 49420, 49526, 49543, 49603, 49627, 49666, 49671, 49722, 49744, 49756, 49766, 49768, 49821, 49827, 49846, 49998, 50062, 50148, 50188, 50200, 50206, 50274, 50317, 50337, 50370, 50423, 50448, 50508, 50655, 50677, 50718, 50724, 50860, 50877, 50954, 50997, 51017, 51062, 51087, 51288, 51447, 51532, 51574, 51624, 51740, 51780, 51821, 51846, 51860, 51905, 51941, 52038, 52040, 52050, 52070, 52130, 52224, 52241, 52378, 52408, 52463, 52477, 52487, 52518, 52527, 52580, 52594, 52607, 52722, 52732, 52852, 52854, 52914, 53046, 53127, 53154, 53340, 53368, 53394, 53424, 53438, 53472, 53486, 53488, 53511, 53538, 53562, 53581, 53631, 53699, 53764, 53820, 54081, 54106, 54175, 54199, 54236, 54296, 54378, 54605, 54660, 54689, 54732, 54822, 54840, 54851, 54881, 54995, 55023, 55049, 55095, 55160, 55230, 55236, 55245, 55303, 55310, 55543, 55553, 55591, 55638, 55716, 55722, 55780, 55802, 55816, 55869, 55886, 56220, 56222, 56244, 56323, 56366, 56442, 56521, 56530, 56547, 56631, 56646, 56656, 56822, 56883, 56919, 57019, 57057, 57079, 57083, 57102, 57124, 57145, 57173, 57241, 57262, 57277, 57347, 57368, 57391, 57439, 57696, 57803, 57861, 57879, 57931, 57970, 58048, 58093, 58148, 58150, 58173, 58230, 58407, 58420, 58451, 58452, 58474, 58575, 58645, 58670, 58877, 58965, 58979, 58987, 58991, 59044, 59056, 59107, 59118, 59162, 59171, 59187, 59228, 59257, 59277, 59301, 59319, 59437, 59475, 59489, 59510, 59518, 59523, 59545, 59601, 59659, 59706, 59727, 59729, 59758, 59824, 59969, 59972, 60035, 60049, 60057, 60058, 60149, 60201, 60309, 60317, 60503, 60554, 60579, 60580, 60582, 60609, 60681, 60710, 60711, 60749, 60857, 60941, 60945, 60967, 61028, 61099, 61138, 61231, 61271, 61298, 61340, 61366, 61388, 61438, 61560, 61643, 61659, 61728, 61903, 62094, 62098, 62104, 62121, 62181, 62263, 62266, 62300, 62350, 62361, 62365, 62377, 62420, 62440, 62455, 62539, 62619, 62625, 62628, 62663, 62681, 62757, 62810, 62927, 62961, 63068, 63069, 63093, 63121, 63133, 63217, 63223, 63317, 63438, 63444, 63449, 63503, 63522, 63547, 63597, 63637, 63655, 63664, 63710, 63731, 63832, 63899, 63963, 63977, 63987, 63996, 64065, 64086, 64140, 64256, 64259, 64273, 64291, 64376, 64424, 64448, 64493, 64558, 64577, 64638, 64645, 64736, 64820, 64841, 64902, 64961, 65066, 65154, 65158, 65171, 65177, 65213, 65239, 65353, 65377, 65429, 65495, 65517, 65597, 65668, 65707, 65731, 65872, 65883, 65981, 65997, 66152, 66238, 66325, 66364, 66367, 66387, 66417, 66483, 66534, 66643, 66679, 66689, 66768, 66863, 66866, 66910, 66915, 66961, 66970, 66987, 67001, 67074, 67253, 67471, 67476, 67501, 67526, 67544, 67575, 67606, 67619, 67635, 67727, 67786, 67886, 67916, 67934, 67970, 68005, 68029, 68060, 68154, 68166, 68393, 68414, 68492, 68532, 68536, 68551, 68664, 68715, 68718, 68725, 68764, 68808, 68873, 68874, 68896, 68907, 68938, 68964, 69014, 69024, 69040, 69077, 69113, 69201, 69232, 69233, 69265, 69287, 69321, 69403, 69427, 69429, 69479, 69564, 69662, 69701, 69712, 69846, 69860, 69862, 69886, 69906, 70143, 70227, 70278, 70466, 70593, 70594, 70640, 70650, 70746, 70926, 70977, 71003, 71011, 71038, 71050, 71131, 71280, 71291, 71293, 71369, 71385, 71642, 71671, 71676, 71734, 71741, 71752, 71774, 71785, 71918, 71928, 71944, 71946, 71978, 72031, 72075, 72088, 72246, 72330, 72348, 72462, 72572, 72576, 72596, 72615, 72637, 72668, 72710, 72712, 72734, 72764, 72879, 72920, 73043, 73186, 73203, 73308, 73329, 73433, 73453, 73489, 73573, 73634, 73689, 73715, 73748, 73812, 73845, 73864, 73893, 73977, 74003, 74016, 74029, 74031, 74131, 74240, 74262, 74296, 74403, 74430, 74463, 74527, 74630, 74634, 74763, 74827, 74922, 74945, 74967, 75000, 75027, 75052, 75053, 75064, 75223, 75299, 75303, 75476, 75484, 75485, 75546, 75591, 75625, 75626, 75787, 75834, 75840, 75892, 75943, 75969, 75997, 76126, 76148, 76153, 76328, 76341, 76358, 76376, 76452, 76496, 76567, 76599, 76631, 76658, 76683, 76779, 76794, 76819, 76843, 76881, 76969, 76984, 77008, 77010, 77047, 77060, 77076, 77158, 77225, 77299, 77316, 77425, 77427, 77450, 77479, 77540, 77559, 77664, 77743, 77775, 77837, 77841, 77988, 78077, 78109, 78151, 78236, 78304, 78405, 78411, 78414, 78444, 78596, 78661, 78665, 78867, 78887, 78928, 78972, 79008, 79033, 79043, 79055, 79062, 79078, 79093, 79142, 79208, 79237, 79341, 79354, 79364, 79400, 79414, 79455, 79484, 79535, 79772, 79857, 80000, 80016, 80029, 80038, 80055, 80088, 80100, 80118, 80176, 80183, 80233, 80241, 80246, 80289, 80301, 80308, 80318, 80326, 80379, 80395, 80404, 80418, 80422, 80464, 80471, 80489, 80494, 80503, 80548, 80580, 80583, 80615, 80667, 80683, 80754, 80839, 80895, 80946, 81031, 81039, 81049, 81093, 81107, 81344, 81381, 81430, 81494, 81510, 81545, 81555, 81605, 81650, 81734, 81787, 81819, 81841, 81903, 81923, 81974, 82084, 82142, 82148, 82241, 82242, 82261, 82354, 82492, 82536, 82552, 82583, 82611, 82654, 82680, 82708, 82745, 82823, 82867, 82968, 82992, 83056, 83124, 83149, 83316, 83394, 83461, 83706, 83730, 83772, 83793, 83877, 83993, 84079, 84107, 84151, 84380, 84420, 84459, 84483, 84585, 84615, 84622, 84632, 84655, 84670, 84763, 84824, 84875, 84909, 84952, 84994, 85000, 85013, 85017, 85125, 85186, 85209, 85219, 85301, 85312, 85374, 85445, 85572, 85600, 85639, 85658, 85676, 85745, 85793, 85841, 85952, 85977, 85998, 86029, 86040, 86099, 86104, 86343, 86368, 86412, 86418, 86421, 86434, 86442, 86522, 86547, 86590, 86645, 86668, 86704, 86717, 86885, 86947, 86992, 87012, 87025, 87090, 87297, 87314, 87332, 87337, 87403, 87421, 87527, 87574, 87737, 87826, 87829, 87870, 87879, 87927, 87953, 88108, 88136, 88137, 88179, 88220, 88241, 88348, 88360, 88446, 88458, 88542, 88665, 88667, 88686, 88719, 88728, 88753, 88776, 88785, 88836, 88994, 89042, 89051, 89220, 89241, 89276, 89352, 89449, 89485, 89580, 89590, 89673, 89757, 89863, 89874, 89904, 89953, 89966, 90001, 90046, 90088, 90098, 90136, 90163, 90238, 90260, 90312, 90338, 90353, 90362, 90418, 90511, 90560, 90578, 90588, 90750, 90784, 90794, 90820, 90844, 90846, 90935, 91050, 91058, 91196, 91218, 91255, 91292, 91308, 91322, 91337, 91406, 91458, 91508, 91510, 91515, 91540, 91615, 91622, 91775, 91830, 92083, 92188, 92305, 92323, 92376, 92378, 92579, 92645, 92681, 92686, 92695, 92797, 92837, 92912, 92927, 92998, 93004, 93035, 93110, 93202, 93208, 93249, 93263, 93281, 93290, 93313, 93328, 93343, 93427, 93446, 93449, 93479, 93823, 93853, 93955, 94086, 94104, 94145, 94147, 94175, 94275, 94288, 94296, 94327, 94344, 94417, 94489, 94497, 94590, 94696, 94711, 94727, 94733, 94770, 94782, 94818, 94828, 94844, 94973, 94996, 94997, 95002, 95005, 95022, 95152, 95181, 95227, 95236, 95241, 95339, 95434, 95435, 95445, 95467, 95532, 95564, 95602, 95621, 95638, 95779, 95791, 95802, 95814, 95892, 95899, 95907, 95988, 96047, 96076, 96124, 96163, 96187, 96240, 96273, 96348, 96355, 96423, 96449, 96463, 96477, 96592, 96619, 96625, 96705, 96720, 96742, 96754, 96770, 96774, 96807, 96910, 96984, 96992, 97001, 97117, 97121, 97133, 97160, 97169, 97190, 97221, 97300, 97379, 97432, 97435, 97490, 97603, 97630, 97681, 97783, 97805, 97821, 97873, 98164, 98195, 98338, 98356, 98409, 98414, 98432, 98447, 98533, 98569, 98639, 98656, 98668, 98705, 98744, 98933, 99037, 99042, 99145, 99179, 99240, 99264, 99287, 99307, 99308, 99351, 99374, 99419, 99420, 99472, 99522, 99536, 99624, 99627, 99710, 99755, 99788, 99808, 99837, 99869, 99888, 99999, 100003, 100064, 100065, 100073, 100107, 100158, 100244])

class TiktokenLmLoader(LMLoader):
    def _tokenize(self):
        multilingual_iters = []
        weights = []

        for data in self.data:
            multilingual_iters.append(
                self._tokenize_foreach_lang(data)
            )
            if 'weight' in data:
                weights.append(float(data['weight']))
            else:
                weights.append(int(data['count']))
        
        if len(multilingual_iters) == 1:
            return multilingual_iters[0]

        sampling_iterator = WeightNoRandomStateIterator(weights, self.seed)
        control_iterator = NativeCheckpointableIterator(sampling_iterator)
        tokenized_lines = iterators.MultiplexIterator(control_iterator, multilingual_iters)
        
        return tokenized_lines

    def _tokenize_foreach_lang(self, data):
        # if 'epoch' in data:
        _random = random.Random(self.seed)
        data_source = data['source']
        epoch_num = 50
        temp_list = math.ceil(epoch_num) * data_source
        _random.shuffle(temp_list)
        dataset = list(zip(temp_list))
        # print('data name: ', data['name'], 'len(dataset): ', len(dataset))
        chunk_files = iterators.ChunkedSourceIterator(
            dataset,
            num_instances=self.num_shards, 
            instance_rank=self.shard_id,)
        # elif self.shuffle:
        #     dataset = list(zip(data['source']))
        #     chunk_files = iterators.InfinitePermutationSourceIterator(
        #         dataset,
        #         seed=self.seed, 
        #         shuffle=self.shuffle, 
        #         num_instances=self.num_shards, 
        #         instance_rank=self.shard_id,)
        # else:
        #     dataset = list(zip(data['source']))
        #     chunk_files = iterators.ChunkedSourceIterator(
        #         dataset,
        #         num_instances=self.num_shards, 
        #         instance_rank=self.shard_id,)
        
        tokenized_lines = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        # tokenized_lines = SelectManyNoSkipIterator(chunk_files, lambda files: self._read_from_files(*files))
        tokenized_lines = iterators.MapIterator(tokenized_lines, self._prepare)
        
        return tokenized_lines

    @staticmethod
    def fs_encode_line(fs_dict, words, append_eos=True):
        ids = []
        for i, word in enumerate(words):
            idx = fs_dict.index(word)
            ids.append(idx)
        if append_eos:
            ids.append(fs_dict.eos_index)
        return ids

    @staticmethod
    def _doc_to_ids(text, spm_tokenizer=None, fs_dict=None):
        assert EOL_SYMBOL in fs_dict.indices

        tokenized_ids = [] # list of list of ids

        assert isinstance(spm_tokenizer, Encoding)
        tokens = spm_tokenizer.encode(text, allowed_special="all")

        offset = 4
        current_list = []
        line_break_flag = False
        for token in tokens:
            if token in TIKTOKEN_LINE_BREAK:
                line_break_flag = True
                current_list.append(token + offset)
            else:
                if line_break_flag:
                    tokenized_ids.append(current_list)
                    current_list = []
                    line_break_flag = False
                current_list.append(token + offset)
        if len(current_list) > 0:
            tokenized_ids.append(current_list)
        tokenized_ids[-1].append(fs_dict.eos_index)
        return tokenized_ids

    def _read_from_files(self, source_file):
        data = []
        if self.args.absolute_path:
            file_path = source_file
        else:
            file_path = os.path.join(self.data_dir, source_file)
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file

        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file

        lines_to_ids = False
        for non_json_key in NON_JSON_SET:
            if non_json_key in file_path:
                lines_to_ids = True
        if lines_to_ids:
            text = "\n".join(lines)
            tokenized_ids = []
            try:
                ret = TiktokenLmLoader._doc_to_ids(text, spm_tokenizer=self.tokenizer, fs_dict=self.dictionary)
                tokenized_ids.extend(ret)
            except BaseException as e:
                print(e)
                print(lines)
        else:
            tokenized_ids = []
            for doc_jsonstr in lines:
                try:
                    json_obj = json.loads(doc_jsonstr)
                    if 'text' in json_obj:
                        text = json_obj['text']
                    elif 'content' in json_obj:
                        text = json_obj['content']
                    if len(text) == 0:
                        continue
                    ret = TiktokenLmLoader._doc_to_ids(text, spm_tokenizer=self.tokenizer, fs_dict=self.dictionary)
                    tokenized_ids.extend(ret)
                except BaseException as e:
                    print(e)
                    print(doc_jsonstr)
            
        # ###################################################

        doc = [self.dictionary.bos()]
        for ids in tokenized_ids:
            if len(doc) + len(ids) > self.tokens_per_sample + 1:
                doc.extend(ids)
                doc = doc[:self.tokens_per_sample + 1]
                data.append(doc)
                doc = [self.dictionary.bos()]
            else:
                doc.extend(ids)

        if len(doc) > 1 and len(doc) <= self.tokens_per_sample + 1:
            data.append(doc)

        return data