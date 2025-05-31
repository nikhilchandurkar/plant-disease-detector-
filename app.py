


from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)
model = torch.load("plant_disease_model.pth", map_location=torch.device("cpu"),weights_only=False)
# model.eval()

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Full class_details dictionary (make sure it's complete with all classes up to model's output)
class_details = {
    0: {
        'name': 'Apple Scab',
        'description': 'Apple Scab appears as dark, velvety patches on both leaves and fruits. These ugly spots start small but grow over time, making apples look bumpy and unappealing. Infected leaves often turn yellow and drop off too early, weakening the entire tree over time.',
        'treatment': 'Start protecting your tree early in spring with fungicides like captan or mancozeb before symptoms appear. Give your tree room to breathe by pruning crowded branches to improve air circulation. Always clean up fallen leaves and fruit from the ground—these harbor the fungus and will reinfect your tree next year if left in place.'
    },
    1: {
        'name': 'Apple Black Rot',
        'description': 'Black Rot creates round, sunken dark spots on apples that expand over time, eventually rotting the fruit completely. On branches, you\'ll notice darkened areas called cankers where the bark looks damaged or dead. These cankers serve as year-round homes for the fungus that slowly weakens your tree.',
        'treatment': 'The most important step is cutting out diseased branches—look for those dark cankers and prune at least 6 inches beyond them into healthy wood. Always disinfect your pruning tools between cuts with rubbing alcohol. Remove all infected fruit and branches from your garden completely. During growing season, apply fungicides to protect healthy tissue and maintain good nutrition to keep tree defenses strong.'
    },
    2: {
        'name': 'Apple Cedar Rust',
        'description': 'This unusual disease creates bright orange-yellow spots on apple leaves that almost look like someone splashed paint on them. In late summer, tiny finger-like tubes may grow from these spots on the leaf undersides. What makes this disease special is that it needs both apple trees AND nearby cedar/juniper trees to complete its life cycle, bouncing back and forth between them.',
        'treatment': 'If possible, remove cedar or juniper trees within 300 feet of your apple trees—these serve as alternative hosts. For existing infections, apply specific rust-preventing fungicides when apple leaves are just developing in spring. Even resistant apple varieties need some care during particularly wet years when disease pressure is high.'
    },
    3: {
        'name': 'Apple (Healthy)',
        'description': 'A healthy apple tree is a beautiful sight! Leaves should be uniformly green, not yellowed or spotted. They should lie flat, not curl or pucker. Fruit develops with smooth, unbroken skin free of spots, bumps, or unusual coloration. Branches should be flexible but strong, with no sunken areas or unusual growths. A robust tree produces new growth each year and holds its leaves until the normal fall drop.',
        'treatment': 'Keep your healthy tree thriving with proper seasonal care. Prune during the dormant season to maintain an open canopy where sunlight and air can reach all branches. Watch for pests during regular inspections, especially in spring and summer. Give your tree balanced nutrition with appropriate fertilizer before bud break, and ensure consistent watering, particularly during fruit development and drought periods.'
    },
    4: {
        'name': 'Blueberry (Healthy)',
        'description': 'Healthy blueberry bushes show vibrant, rich green leaves (sometimes with a blue-green tint) that are firm and smooth. New growth appears reddish before maturing to green. Stems should be flexible and covered with smooth bark. During season, berries develop with a consistent deep blue color covered by a natural whitish "bloom" powder. The plant maintains steady growth with no wilting, spotting, or unusual discoloration.',
        'treatment': 'Keep your blueberries thriving by maintaining soil acidity—these plants love sour soil with pH between 4.5–5.5, much more acidic than most garden plants. Water consistently but never let roots sit in soggy conditions. Apply a specialized acid-loving plant fertilizer in early spring, and add pine needle or pine bark mulch to slowly increase acidity while retaining moisture. Check regularly for the earliest signs of pests or disease.'
    },
    5: {
        'name': 'Cherry Powdery Mildew',
        'description': 'This fungal disease looks exactly like its name suggests—a whitish, powder-like coating that appears on cherry leaves, stems, and sometimes fruit. Leaves may curl upward and show stunted growth. The white coating can be rubbed off with your fingers, unlike some diseases that stain the leaf tissue. In severe cases, the infection prevents fruit from developing properly and can weaken the entire tree over several seasons.',
        'treatment': 'At the first sign of white powder on leaves, begin treatment with sulfur-based fungicides or more powerful systemic options for severe cases. Improve air movement throughout your tree by thinning interior branches and removing "water sprouts" (fast-growing vertical shoots). Always clean up fallen leaves promptly as they can harbor the fungus. Water at the base rather than overhead to keep foliage dry whenever possible.'
    },
    6: {
        'name': 'Cherry (Healthy)',
        'description': 'A thriving cherry tree showcases glossy, deep green leaves arranged evenly along branches. The leaves lie flat and smooth without curling or puckering. New growth appears vigorous with consistent coloring. When fruiting, healthy cherries grow in abundant clusters, developing their characteristic color—whether bright red, dark burgundy, or golden yellow—evenly across the fruit. Bark appears smooth on young branches with no oozing, cracking, or unusual bumps.',
        'treatment': 'Maintain your cherry tree\'s good health with deep, infrequent watering that reaches the entire root zone. Apply balanced fertilizer in early spring before leaf emergence. Perform yearly pruning during the dormant season to remove any crossing or crowded branches, creating an open, vase-like structure that allows sunlight to reach all parts of the tree. Regular care minimizes the risk of future disease problems.'
    },
    7: {
        'name': 'Corn Cercospora Leaf Spot',
        'description': 'This fungal disease creates small, grayish spots scattered across corn leaves. Each spot typically has a defined border with a pale center and darker edge, sometimes with a yellowing halo around it. Multiple spots can merge together over time. As the disease progresses, infected leaves may turn brown and die prematurely. The damaged leaves can\'t photosynthesize properly, which reduces energy available for ear development and lowers your harvest quality.',
        'treatment': 'When planning your garden, look for corn varieties labeled as disease-resistant to Cercospora. Don\'t plant corn in the same location year after year—rotate crops to different garden areas to prevent fungus buildup in soil. During humid or rainy periods when disease is most likely to spread, apply appropriate fungicides to protect healthy leaves. Remove severely infected lower leaves to slow disease progression to the upper plant.'
    },
    8: {
        'name': 'Corn Common Rust',
        'description': 'Common rust is easy to identify—look for raised, reddish-brown bumps scattered across corn leaves. These pustules feel rough to the touch and may release an orange-red powdery substance (spores) when rubbed. The disease typically starts on lower leaves and moves upward as spores spread. While it rarely kills plants completely, heavy infections steal energy from ear formation, reducing your harvest size and quality.',
        'treatment': 'Choose corn varieties with built-in rust resistance whenever possible. Monitor your corn patch regularly, especially during humid weather when rust thrives. At the first sign of rust pustules, apply a fungicide labeled for corn rust. Practice good garden sanitation by removing corn debris after harvest rather than leaving it in place. Good air circulation helps too, so avoid overcrowded plantings.'
    },
    9: {
        'name': 'Corn Northern Leaf Blight',
        'description': 'Northern leaf blight creates distinctive long, cigar-shaped gray or tan lesions on corn leaves, each potentially several inches long. These large spots often appear first on lower leaves after rainy periods, then spread upward through the plant. Severely infected leaves wither and die prematurely. When infection happens early in the growing season or spreads to the leaves near the developing ear, yield losses can be significant.',
        'treatment': 'Start with resistant corn varieties—seed packages or descriptions often list specific disease resistances. Scout your plants regularly, especially during humid weather conditions that favor disease development. Apply recommended fungicides at the first sign of infection, focusing on protecting upper leaves that feed the developing ears. After harvest, till under or remove old corn debris, as the fungus overwinters in infected plant material.'
    },
    10: {
        'name': 'Corn (Healthy)',
        'description': 'Healthy corn stands tall with thick, sturdy stalks and broad, dark green leaves that point slightly upward before gracefully arching over at the tips. Leaves should have consistent coloring without spots, stripes, or yellowing. The plant grows vigorously, adding new leaves from the center as it matures. Tassels (the male flowers at the top) develop fully without browning, while silks emerge from developing ears with a fresh, light green or yellowish appearance. Well-grown corn develops full ears with tightly packed kernels.',
        'treatment': 'Support corn\'s heavy nutrient needs with balanced fertilizer, applying extra nitrogen when plants are knee-high. Plant in blocks rather than single rows for better pollination and full kernel development. Consistent moisture is crucial, particularly during tasseling and ear formation—drought stress during these periods can significantly reduce yield. Regular monitoring catches pests early when they\'re easier to manage.'
    },
    11: {
        'name': 'Grape Black Rot',
        'description': 'Black rot is devastating for grape growers. It begins with small yellowish spots on leaves that grow into brown lesions with dark borders. The real damage happens on the fruit—infected grapes develop whitish dots that rapidly expand. The berries then shrivel and turn into hard, black, mummified fruits that remain attached to the vine. These "mummies" spread infection to next season\'s growth if not removed.',
        'treatment': 'Sanitation is crucial—remove all mummified berries from vines and from the ground beneath plants. Prune vines to improve airflow, focusing on opening the canopy to help leaves dry quickly after rain or morning dew. Begin fungicide applications early in the season (just after bud break) and continue at recommended intervals throughout the growing season, especially during rainy periods when spores spread easily.'
    },
    12: {
        'name': 'Grape Esca (Black Measles)',
        'description': 'This complex disease, sometimes called "black measles," affects grapevines from the inside out. The most visible symptoms include leaves with striking yellow-red-brown discoloration between the veins, making a tiger-stripe pattern. Berries develop purple-black spots resembling measles, and may crack or shrivel instead of ripening normally. Inside the woody parts of the vine, you\'ll find dark streaking when branches are cut. This chronic disease slowly weakens and eventually kills vines over several seasons.',
        'treatment': 'Unfortunately, there is no complete cure for Esca once a vine is infected. Careful pruning practices are essential—always disinfect tools between cuts and avoid pruning during wet weather when fungal spores spread easily. Remove heavily infected vines completely. For less affected vines, cut away discolored wood until you reach clean tissue. Some growers apply wound protectants to large pruning cuts to prevent fungal entry. Maintain overall vine vigor through proper nutrition and water management.'
    },
    13: {
        'name': 'Grape Leaf Blight',
        'description': 'Grape leaf blight creates irregularly shaped brown patches on leaves that may be surrounded by a yellow halo. As these spots enlarge and merge, entire sections of leaves die and turn papery. In severe cases, premature defoliation (leaf drop) occurs, exposing fruit to sunburn and preventing proper ripening. The disease spreads quickly during warm, humid conditions, especially when leaves remain wet for extended periods.',
        'treatment': 'Improve air circulation by proper pruning and training of vines, allowing leaves to dry quickly after rain or morning dew. Apply copper-based fungicides early in the growing season before symptoms appear. When watering, avoid wetting the foliage—use drip irrigation instead of overhead sprinklers. Clear away fallen leaves and other plant debris regularly, as these can harbor the fungus and reinfect plants.'
    },
    14: {
        'name': 'Grape (Healthy)',
        'description': 'Healthy grapevines showcase vibrant green leaves with well-defined lobes and serrated edges. New shoots appear straight and strong with consistent coloration. Leaf arrangement along the vine allows for good light penetration throughout the canopy. Developing grape clusters show uniform berry size and color appropriate to their variety and ripening stage. The vine demonstrates steady growth throughout the season without weak or dying sections.',
        'treatment': 'Maintain vine health with regular pruning to control growth and open the canopy for good air circulation and light exposure. Apply balanced fertilizer according to soil test recommendations, avoiding excessive nitrogen which promotes leafy growth at the expense of fruit production. Establish consistent watering patterns that provide deep moisture without creating constantly wet conditions that promote disease. Scout regularly for early signs of common grape pests like leafhoppers or mites.'
    },
    15: {
        'name': 'Orange Huanglongbing (Citrus Greening)',
        'description': 'This devastating bacterial disease, spread by tiny insects called citrus psyllids, is a serious threat to orange and other citrus trees. Leaves develop blotchy, asymmetrical yellow mottling (not the same on both halves of the leaf) and often appear twisted or smaller than normal. Fruit remains partially green even when ripe, develops a bitter taste, and is often small and lopsided. The disease gradually kills the tree over several years as it blocks nutrient flow within the plant.',
        'treatment': 'There is no cure once a tree is infected, making prevention critical. Control the psyllid insects that spread the disease using appropriate insecticides or horticultural oils. Inspect trees frequently for signs of psyllids or symptoms. If a tree shows clear symptoms, it should be removed completely to prevent spreading the disease to healthy trees. When planting new citrus, always purchase certified disease-free trees from reputable nurseries rather than moving plants from unknown sources.'
    },
    16: {
        'name': 'Peach Bacterial Spot',
        'description': 'This bacterial infection creates small, angular water-soaked spots on peach leaves that later turn purple or brown and may fall out, giving a "shothole" appearance. On fruit, look for small sunken spots that may crack open as the fruit grows. These lesions make the peaches susceptible to other rots and destroy their market value. The bacteria enter through natural openings and wounds, especially during wet, humid conditions.',
        'treatment': 'Plant resistant peach varieties when establishing new trees—some varieties show much better tolerance than others. Apply copper-based bactericides beginning at leaf drop in fall and continuing through early spring before bud break. Be careful with copper applications during the growing season as they can damage leaves and fruit if applied incorrectly. Prune during dry weather to avoid spreading the bacteria through pruning wounds.'
    },
    17: {
        'name': 'Peach (Healthy)',
        'description': 'A thriving peach tree displays lance-shaped leaves with a uniform medium to dark green color and finely serrated edges. Leaves are spaced well along branches without crowding. During the growing season, new growth appears vigorous with no spotting or curling. Fruit develops with an even shape and gradually gains its characteristic blush as it ripens. The bark appears clean without oozing, cracks, or raised bumps that might indicate pest problems.',
        'treatment': 'Keep peach trees vigorous with annual pruning to maintain an open, vase-shaped structure that allows light and air to reach all branches. Thin developing fruit when they reach marble-size so remaining peaches are spaced about 6-8 inches apart—this improves size and quality. Deep, infrequent watering encourages strong root development, while mulching helps retain soil moisture. Regular feeding with balanced fertilizer supports healthy growth and fruit production.'
    },
    18: {
        'name': 'Bell Pepper Bacterial Spot',
        'description': 'This troublesome bacterial disease creates small, water-soaked spots on pepper leaves that enlarge and turn brown, often with a yellowish halo. On the peppers themselves, raised scabby spots develop, making them unmarketable. The disease spreads rapidly during warm, wet weather, moving through the garden by water splash, tools, or handling. Severely affected plants drop their leaves and stop producing usable fruit.',
        'treatment': 'Prevention is key—avoid overhead watering that keeps leaves wet and spreads bacteria. Instead, use drip irrigation or soaker hoses that deliver water directly to the soil. Apply copper-based products preventively before symptoms appear, especially during humid weather when disease pressure is high. In vegetable gardens, practice crop rotation by not planting peppers, tomatoes, or eggplants (all related crops) in the same spot for at least three years.'
    },
    19: {
        'name': 'Bell Pepper (Healthy)',
        'description': 'Healthy bell pepper plants stand upright with strong stems supporting abundant, broad leaves of consistent green color. The leaf surface appears smooth without spots, holes, or yellowing. Flowers form and set fruit normally, with peppers developing their appropriate shape without deformities or blemishes. As peppers mature, they size up evenly, changing from green to their ripe color (red, yellow, orange, etc.) with glossy, smooth skin.',
        'treatment': 'Support pepper plants with consistent care throughout the growing season. Provide even moisture—fluctuations can cause problems like blossom end rot. Apply balanced fertilizer when planting and again when fruits begin to form. Many gardeners stake or cage larger pepper varieties to prevent branches from breaking under the weight of heavy fruit. Maintain good spacing between plants to ensure adequate airflow that discourages disease development.'
    },
    20: {
        'name': 'Potato Early Blight',
        'description': 'Early blight creates distinctive target-like spots on potato leaves—dark brown rings with a pattern resembling a bullseye. The disease typically starts on older, lower leaves and moves upward. As the spots enlarge and multiply, affected leaves yellow and die prematurely. This reduces the plant\'s ability to produce energy, ultimately resulting in smaller potatoes and lower yields. The fungus can also cause dry, dark lesions on tubers themselves.',
        'treatment': 'Rotate potato plantings so they don\'t grow in the same spot more than once every three years—the fungus persists in soil. Apply mulch to prevent soil (and the fungus it contains) from splashing onto lower leaves. At the first sign of symptoms, apply appropriate fungicides like chlorothalonil, followed by regular applications according to label directions. Remove and dispose of severely infected leaves to slow disease spread.'
    },
    21: {
        'name': 'Potato Late Blight',
        'description': 'The infamous culprit behind the Irish Potato Famine, late blight remains one of the most destructive plant diseases. It creates large, irregular dark patches on leaves, often with fuzzy white growth visible on the undersides during humid conditions. Stems develop dark lesions, and infected tubers show reddish-brown patches externally and internally. Once established, the disease can destroy an entire potato crop in days under favorable cool, wet conditions.',
        'treatment': 'This aggressive disease requires immediate action. Remove and destroy infected plants entirely—do not compost them. Apply preventive fungicides before symptoms appear when conditions favor disease development (cool, wet weather). For home gardeners, consider covering plants with row covers during extended rainy periods. Plant certified disease-free seed potatoes, and hill soil around stems as they grow to protect developing tubers.'
    },
    22: {
        'name': 'Potato (Healthy)',
        'description': 'Healthy potato plants display lush green foliage with compound leaves made up of several leaflets. Stems stand strong and upright without discoloration or lesions. New growth emerges vigorously, and if flowering, the plant produces small white to purple blossoms. Below ground, tubers develop with clean skins free of scab, cracks, or discoloration. A robust potato plant maintains its foliage until natural senescence late in the season.',
        'treatment': 'Ensure potato success by starting with certified disease-free seed potatoes rather than grocery store potatoes that may carry diseases. Plant in well-drained soil amended with organic matter. Hill soil around stems as plants grow to prevent tubers from being exposed to light (which causes greening and toxic compounds). Keep weeds controlled as they compete for nutrients and can harbor pests. Water deeply but avoid constant wet conditions that promote rot and disease.'
    },
    23: {
        'name': 'Strawberry Leaf Scorch',
        'description': 'This fungal disease begins with small, purple spots on strawberry leaves that expand into larger, irregularly shaped areas. The distinctive feature is how the leaf edges and the tissue between veins turns red, purple, or brown, giving a scorched appearance—as if the leaves were burned at the edges. Severely infected leaves dry up completely and die. The disease reduces the plant\'s vigor and dramatically impacts fruit production over time.',
        'treatment': 'Begin treatment by removing and disposing of affected leaves at the first sign of infection. Improve air circulation by thinning plants and removing weeds. Avoid overhead watering that keeps leaves wet—use drip irrigation instead. Water early in the day so any wet leaves can dry completely before evening. Apply fungicides labeled for strawberry leaf diseases according to package directions. After fruiting, renovate beds by mowing off foliage (if not severely diseased) to encourage healthy new growth.'
    },
    24: {
        'name': 'Strawberry (Healthy)',
        'description': 'Thriving strawberry plants feature deep green, trifoliate (three-part) leaves with serrated edges arranged in compact rosettes. New leaves emerge from the center with proper unfurling and coloration. When flowering, healthy plants produce white blossoms with yellow centers on strong stems held above the foliage. As berries develop, they show consistent shape and color development, gradually turning from green to white to their full ripe color without spots or deformities.',
        'treatment': 'Maintain strawberry patch health by providing consistent moisture—never letting plants completely dry out, but also avoiding waterlogged conditions that promote root diseases. Apply balanced fertilizer in early spring as growth begins and again after the main harvest. Mulch around plants with clean straw or pine needles to keep berries off soil, reduce weed competition, and conserve moisture. Replace plants every 3-4 years as older plants become less productive and more susceptible to disease.'
    },
    25: {
        'name': 'Tomato Bacterial Spot',
        'description': 'This bacterial infection creates small, dark, water-soaked spots on tomato leaves, stems, and fruit. On leaves, spots may have a yellow halo and eventually dry out, sometimes causing holes as the dead tissue falls out. Fruit spots start small but develop into raised, scabby areas that make tomatoes unappealing and vulnerable to rot. The disease spreads rapidly in warm, wet weather and can significantly reduce harvest quality and quantity.',
        'treatment': 'Start with disease-free seeds and transplants from reputable sources. Apply copper-based sprays early in the season before symptoms appear, especially during rainy periods. Avoid working with plants when they\'re wet, as this can spread bacteria from plant to plant. Practice crop rotation, avoiding planting tomatoes, peppers, or eggplants in the same spot for at least three years. Remove and destroy severely infected plants to prevent spreading to healthy ones.'
    },
    26: {
        'name': 'Tomato Early Blight',
        'description': 'Early blight is recognizable by distinct dark brown spots with concentric rings that create a target-like or bullseye pattern on tomato leaves. The disease typically begins on older, lower leaves and moves upward as spores splash onto healthy tissue. Affected leaves eventually yellow and drop off. Stem lesions may form as dark, sunken areas, and fruit can develop leathery, sunken spots usually near the stem end. The progressive loss of leaves reduces fruit production and quality.',
        'treatment': 'Apply fungicides containing chlorothalonil or copper early in the season before symptoms appear, continuing applications through the growing season according to label instructions. Stake or cage plants to keep foliage off the ground and improve air circulation. Mulch around plants to prevent soil splashing onto leaves during rain or irrigation. Water at the base of plants rather than overhead to keep foliage dry. Remove and destroy badly infected leaves promptly.'
    },
    27: {
        'name': 'Tomato Late Blight',
        'description': 'Late blight is one of the most destructive tomato diseases, capable of destroying entire plants within days under favorable conditions. Look for large, irregular, water-soaked patches on leaves that quickly turn brown or black. During humid conditions, white fungal growth may be visible on the underside of leaves. Stems develop dark lesions, and fruit shows large, firm brown areas. This aggressive disease spreads extremely rapidly during cool, wet weather.',
        'treatment': 'Act immediately upon first symptoms—remove and bag infected plants to prevent spores from spreading to healthy plants. Do not compost infected material. Apply fungicides preventatively before symptoms appear when weather conditions favor disease development (cool temperatures with high humidity). For protection, use fungicides containing chlorothalonil, mancozeb, or copper. Consider growing late blight resistant varieties if the disease is common in your area.'
    },
    28: {
        'name': 'Tomato Leaf Mold',
        'description': 'Primarily a greenhouse disease but also problematic in humid gardens, leaf mold causes pale green or yellow patches on the upper leaf surface. The diagnostic feature is the velvety, olive-green to grayish-purple fungal growth that develops on the undersides of these spots. As the disease progresses, leaves may curl, wither, and drop prematurely. Fruit infection is rare but can occur. The disease thrives in environments with high humidity and poor air circulation.',
        'treatment': 'Improve growing conditions by increasing spacing between plants and using fans in greenhouses to improve air circulation. Reduce humidity by avoiding overhead watering and watering early in the day so leaves can dry before evening. Remove and destroy infected leaves promptly. Apply fungicides containing chlorothalonil when symptoms first appear, especially if growing conditions can\'t be optimized. For greenhouse production, consider resistant varieties specifically developed for protected cultivation.'
    },
    29: {
        'name': 'Tomato Septoria Leaf Spot',
        'description': 'This common fungal disease creates numerous small, circular spots with dark borders and lighter gray or tan centers on tomato leaves. Unlike early blight, these spots don\'t form concentric rings or target patterns. The disease typically starts on lower leaves and moves upward. Severely affected leaves turn yellow, then brown, and fall off. While the fungus rarely infects stems or fruit directly, the loss of foliage exposes fruit to sunscald and reduces plant productivity.',
        'treatment': 'Remove infected leaves as soon as you notice them to slow disease spread. Apply mulch around plants to prevent soil splash, which can carry fungal spores from the ground to leaves. Use fungicides containing chlorothalonil, copper, or mancozeb at first sign of disease, applying according to label directions. Ensure adequate spacing between plants for good air circulation. At season\'s end, thoroughly clean up all plant debris as the fungus overwinters on old tomato material.'
    },
    30: {
        'name': 'Tomato Spider Mites',
        'description': 'Despite their name, spider mites aren\'t actually insects but tiny arachnids barely visible to the naked eye. Their feeding causes distinctive yellow stippling or speckling on leaf surfaces—almost like someone sprinkled tiny yellow dots across the leaves. As infestations progress, leaves take on a bronzed or dusty appearance, and fine webbing may appear, especially between leaf stems and at growing tips. Severe infestations cause leaves to dry up and fall off, reducing plant vigor and yield.',
        'treatment': 'First, try simple physical controls: spray plants forcefully with water to dislodge mites, focusing on leaf undersides where they hide. Spider mites thrive in hot, dry conditions, so increasing humidity around plants can discourage them. Apply insecticidal soap or neem oil, thoroughly covering both sides of leaves. For severe infestations, consider miticides specifically labeled for vegetable garden use. Avoid broad-spectrum insecticides that kill beneficial predators that naturally control mite populations.'
    },
    31: {
        'name': 'Tomato Target Spot',
        'description': 'Target spot creates distinctive circular lesions with concentric rings and a lighter center—truly resembling a target or bullseye. These spots appear on leaves, stems, and fruit. As the disease progresses, leaf spots may merge, causing extensive tissue death and premature leaf drop. On fruit, target spot begins as small, dark, sunken spots that expand and develop concentric rings, making tomatoes unmarketable. The disease spreads rapidly during warm, wet conditions.',
        'treatment': 'Start by improving plant spacing to ensure good air circulation that helps leaves dry quickly after rain or morning dew. Apply mulch to reduce soil splash that can spread fungal spores from the ground to leaves. Use stakes or cages to keep plants upright and foliage off the ground. Apply fungicides labeled for target spot at the first sign of disease, focusing on good coverage of both upper and lower leaf surfaces. Remove severely infected leaves and destroy rather than composting them.'
    },
    32: {
        'name': 'Tomato Yellow Leaf Curl Virus',
        'description': 'This devastating viral disease creates a distinctive appearance: leaves curl upward and inward, becoming thick and rubbery with a yellow margin. New leaves emerge already deformed and stunted. Plants infected when young remain severely stunted and produce few if any usable tomatoes. The virus is spread by tiny insects called whiteflies—small white flying insects that gather on the undersides of leaves. Once a plant is infected, there is no cure.',
        'treatment': 'Prevention is the only effective strategy. Control whiteflies aggressively using insecticidal soaps, neem oil, or yellow sticky traps. Cover young plants with fine mesh row covers to exclude whiteflies, removing covers temporarily during flowering for pollination. Plant virus-resistant tomato varieties when available—seed catalogs typically indicate resistance with "TYLCV" in the description. Remove and destroy infected plants immediately to prevent whiteflies from spreading the virus to healthy plants.'
    },
    33: {
        'name': 'Tomato Mosaic Virus',
        'description': 'This persistent viral disease creates a mottled pattern of light and dark green or yellow and green on tomato leaves—hence the "mosaic" name. Leaves may also appear distorted, curled, or smaller than normal. Plant growth becomes stunted and fruit production significantly reduced. On fruit, the virus can cause uneven ripening or internal browning. The virus spreads easily through direct plant contact, contaminated tools, and even on smokers\' hands from tobacco products (which can carry related viruses).',
        'treatment': 'There is no cure for infected plants, which should be removed and destroyed promptly. Prevention is essential: wash hands thoroughly with soap before handling plants, especially after smoking. Disinfect garden tools with a 10% bleach solution or rubbing alcohol between uses. Plant resistant varieties indicated by "TMV" or "ToMV" in seed descriptions. Control weeds around the garden as they can harbor the virus. Always buy certified disease-free seeds and transplants from reputable sources.'
    },
    34: {
        'name': 'Tomato (Healthy)',
        'description': 'A thriving tomato plant shows vibrant green leaves that are slightly fuzzy to the touch and have a distinctive tomato smell when brushed. Stems appear sturdy and strong enough to support the weight of developing fruit. New growth emerges normally from the tips and leaf axils. Flowers form in clusters and have bright yellow petals that open fully. As fruits develop, they size up evenly with smooth skin, gradually changing from green to their mature color (red, yellow, purple, etc.) with consistent coloration.',
        'treatment': 'Support your tomatoes throughout the season with proper care. Provide strong stakes or cages to keep heavy fruit off the ground. Water deeply but infrequently at the base of plants, aiming for 1-2 inches per week. Apply organic mulch to conserve moisture and suppress weeds. Feed with balanced fertilizer when planting and again when first fruits begin to develop. Regularly inspect for early signs of common pests like hornworms or diseases—early detection makes management much easier.'
    },
    35: {
        'name': 'Soybean (Healthy)',
        'description': 'Healthy soybean plants have trifoliate (three-part) leaves of medium to dark green with smooth edges. Plants stand upright with strong stems and branches. Growth appears uniform across the field or garden, with new leaves emerging consistently. During flowering, small white or purple blossoms develop along stems, followed by small fuzzy pods that gradually fill out as the beans develop inside. Leaves maintain their green color until the natural yellowing that occurs as plants begin to mature.',
        'treatment': 'Support soybean growth with proper field preparation, including good drainage—soybeans dislike "wet feet." Plant when soil temperatures reach at least 60°F for optimal germination. Keep fields well-weeded, especially during early growth stages when young plants compete poorly with weeds. Scout regularly for common pests like aphids, which can build up quickly during favorable conditions. Ensure adequate moisture during pod filling stages for maximum yield.'
    },
    
    36: {
        'name': 'Squash Powdery Mildew',
        'description': 'This common fungal disease appears as a white, powdery coating on the upper and sometimes lower surfaces of squash leaves. It begins as small, circular powdery spots that expand and merge to cover entire leaf surfaces. As the disease progresses, leaves may yellow, brown, and die prematurely. The white powder can also appear on stems and leaf stalks. While powdery mildew rarely kills plants outright, it significantly reduces vigor, yield, and fruit quality by decreasing the plant\'s ability to photosynthesize.',
        'treatment': 'Improve air circulation by proper plant spacing and selective pruning of dense growth. Water at the base of plants rather than overhead to keep foliage dry. Apply sulfur-based fungicides or potassium bicarbonate (baking soda) sprays at first detection. For organic options, try milk sprays (1 part milk to 9 parts water) or neem oil. Remove severely infected leaves and clean up all plant debris at season\'s end to reduce fungal spores that could overwinter.'
    },
    37: {
        'name': 'Wheat Leaf Rust',
        'description': 'Leaf rust creates small, oval-shaped, raised orange-red pustules scattered across wheat leaves. These pustules contain thousands of spores that give them their rusty coloration and powdery texture. The disease typically starts on lower leaves and progresses upward. Heavy infections cause leaves to yellow and die prematurely, reducing the plant\'s ability to produce and fill grain. Under severe pressure, wheat yields can be reduced by 40% or more.',
        'treatment': 'Choose wheat varieties with genetic resistance to leaf rust—this is the most effective and economical control method. Apply appropriate fungicides during key growth stages (typically around flag leaf emergence) for susceptible varieties. Timing is critical, as applications protect new growth from infection. For backyard or small-scale growers, remove volunteer wheat plants that may harbor the disease between seasons, and practice crop rotation rather than planting wheat in the same location year after year.'
    },
    38: {
        'name': 'Wheat (Healthy)',
        'description': 'Healthy wheat plants grow in an upright form with strong stems and evenly spaced tillers (side shoots). Leaves appear uniformly green, narrow, and blade-like without spots or discoloration. As plants mature, seed heads develop at the top with a characteristic shape based on variety (could be awned or awnless). The heads fill with plump kernels that mature from green to golden brown. Overall, the crop shows consistent height and development across the field, free from lodging (falling over) or irregular growth patterns.',
        'treatment': 'Support wheat growth with proper soil preparation and fertility—conduct soil tests to determine specific nutrient needs. Plant at the optimal time for your region, as timing significantly affects yield potential. Control weeds early when competition can most damage developing wheat plants. Scout fields regularly during the growing season to catch any disease or pest issues before they become severe. Harvest at the proper moisture content to maximize grain quality and storage potential.'
    },
    39: {
        'name': 'Unknown or Other Class',
        'description': 'This classification appears when the system encounters a plant condition it hasn\'t been trained to recognize. The image might show an unusual disease symptom, a pest problem rather than a disease, an environmental stress condition (like drought or nutrient deficiency), or possibly a new disease that wasn\'t included in the original training data. Alternatively, the image quality or composition might make accurate classification difficult.',
        'treatment': 'First, try taking clearer, well-lit photos that focus specifically on the affected plant parts from multiple angles. If using the system for diagnosis, consider consulting with local agricultural extension services who can provide region-specific expertise. For researchers or system developers, this classification might indicate areas where additional training data would be valuable to improve the system\'s recognition capabilities.'
    }
}



@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    disease_info = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            image = Image.open(path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                predicted_index = output.argmax().item()
                disease_info = class_details.get(predicted_index, {
                    "name": "Unknown",
                    "description": "Not found in database.",
                    "treatment": "Try uploading a clearer image or consult an expert."
                })
                prediction = disease_info["name"]
                image_url = path

    return render_template("index.html", prediction=prediction, disease_info=disease_info, image_url=image_url)


if __name__ == "__main__":
    app.run(debug=True)
