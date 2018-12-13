#coding=utf-8
#扩展标签较少的图片
import numpy as np
import random
IMAGE_LABEL_FILE="/home/niusilong/svn/svn_repository_ai/competition_inception_v4/label_file/4740_absolute_path_gt30.txt"
LABEL_ID_DICT={1:"1-turn(转弯)",2:"2-land(土地)",3:"3-albatross(信天翁)",4:"4-f-16(f-16)",5:"5-bengal(孟加拉虎)",6:"6-shrubs(灌木)",7:"7-valley(山谷)",8:"8-mare(母马)",9:"9-monastery(修道院)",10:"10-palace(宫)",11:"11-grizzly(灰熊)",12:"12-bear(熊)",13:"13-flag(国旗)",14:"14-pillar(柱子)",15:"15-shops(商店)",16:"16-canal(运河)",17:"17-cars(汽车)",18:"18-doorway(门口)",19:"19-sky(天空)",20:"20-kit(系列)",21:"21-vendor(小贩)",22:"22-interior(室内)",23:"23-column(柱子)",24:"24-lake(湖)",25:"25-buildings(建筑)",26:"26-baby(婴儿)",27:"27-hawaii(夏威夷)",28:"28-road(路)",29:"29-birds(鸟)",30:"30-fan(扇贝)",31:"31-cow(牛)",32:"32-sheep(羊)",33:"33-basket(篮子)",34:"34-marine(海洋)",35:"35-coral(珊瑚)",36:"36-dress(衣服)",37:"37-jet(喷气式飞机)",38:"38-roofs(屋顶)",39:"39-white-tailed(白尾鹿)",40:"40-forest(森林)",41:"41-landscape(景观)",42:"42-restaurant(餐厅)",43:"43-shadows(阴影)",44:"44-cubs(幼崽)",45:"45-close-up(特写镜头)",46:"46-gate(门)",47:"47-fly(飞)",48:"48-ruins(废墟)",49:"49-stone(石头)",50:"50-flight(飞行器)",51:"51-sidewalk(人行道上)",52:"52-store(商店)",53:"53-tulip(郁金香)",54:"54-sea(海)",55:"55-moose(驼鹿)",56:"56-tusks(象牙)",57:"57-oahu(瓦胡岛)",58:"58-clouds(云)",59:"59-temple(寺庙)",60:"60-prop(道具)",61:"61-tower(塔)",62:"62-sails(帆)",63:"63-caribou(驯鹿)",64:"64-stems(茎)",65:"65-dunes(沙丘)",66:"66-art(艺术)",67:"67-rocks(岩石)",68:"68-sunrise(日出)",69:"69-tundra(苔原)",70:"70-silhouette(轮廓)",71:"71-smoke(烟)",72:"72-water(水)",73:"73-reptile(爬行动物)",74:"74-city(城市)",75:"75-porcupine(豪猪)",76:"76-woman(女人)",77:"77-shore(海岸)",78:"78-balcony(阳台)",79:"79-clothes(衣服)",80:"80-mountain(山)",81:"81-tree(树)",82:"82-elephant(大象)",83:"83-trunk(象鼻)",84:"84-horizon(地平线)",85:"85-train(火车)",86:"86-courtyard(院子里)",87:"87-vineyard(葡萄园)",88:"88-bulls(公牛)",89:"89-entrance(入口)",90:"90-path(路径)",91:"91-hillside(山坡上)",92:"92-maui(毛伊岛)",93:"93-sculpture(雕塑)",94:"94-wall(墙)",95:"95-face(脸)",96:"96-food(食物)",97:"97-fountain(喷泉)",98:"98-window(窗口)",99:"99-sunset(日落)",100:"100-peaks(山峰)",101:"101-tiger(老虎)",102:"102-snake(蛇)",103:"103-log(日志)",104:"104-crab(螃蟹)",105:"105-branch(树枝)",106:"106-field(场)",107:"107-tails(尾巴)",108:"108-canyon(峡谷)",109:"109-cottage(小屋)",110:"110-slope(坡)",111:"111-plane(飞机)",112:"112-petals(花瓣)",113:"113-relief(浮雕)",114:"114-mist(雾)",115:"115-lynx(山猫)",116:"116-pyramid(金字塔)",117:"117-palm(棕榈)",118:"118-meadow(草地)",119:"119-mosque(清真寺)",120:"120-blooms(花朵)",121:"121-architecture(建筑风格)",122:"122-market(市场)",123:"123-crystals(晶体)",124:"124-waves(波浪)",125:"125-lawn(草坪上)",126:"126-ground(地面)",127:"127-locomotive(火车头)",128:"128-lion(狮子)",129:"129-mule(骡马)",130:"130-den(兽穴)",131:"131-coyote(土狼)",132:"132-ships(船只)",133:"133-marsh(沼泽)",134:"134-booby(塘鹅)",135:"135-kauai(考艾岛)",136:"136-hats(帽子)",137:"137-nets(网)",138:"138-anemone(海葵)",139:"139-fence(栅栏)",140:"140-african(非洲)",141:"141-outside(外)",142:"142-cathedral(大教堂)",143:"143-runway(跑道)",144:"144-vines(葡萄树)",145:"145-buddha(佛)",146:"146-desert(沙漠)",147:"147-bush(灌木)",148:"148-calf(小牛)",149:"149-crafts(工艺品)",150:"150-indian(印度)",151:"151-village(村)",152:"152-formula(方程式赛车)",153:"153-squirrel(松鼠)",154:"154-needles(针)",155:"155-formation(编队)",156:"156-detail(细节)",157:"157-buddhist(佛教)",158:"158-sign(标志)",159:"159-lighthouse(灯塔)",160:"160-foals(小马驹)",161:"161-herd(兽群)",162:"162-costume(服装)",163:"163-bridge(桥)",164:"164-man(男人)",165:"165-reefs(珊瑚礁)",166:"166-horns(角)",167:"167-night(晚上)",168:"168-reflection(反射)",169:"169-cougar(美洲狮)",170:"170-light(光)",171:"171-fruit(水果)",172:"172-horses(马)",173:"173-pool(池)",174:"174-zebra(斑马)",175:"175-street(街)",176:"176-vegetation(植被)",177:"177-decoration(装饰)",178:"178-tables(表)",179:"179-terrace(阳台)",180:"180-statue(雕像)",181:"181-grass(草)",182:"182-sphinx(斯芬克斯)",183:"183-arctic(北极)",184:"184-boats(船)",185:"185-coast(海岸)",186:"186-post(帖子)",187:"187-black(黑色的)",188:"188-fish(鱼)",189:"189-skyline(天际线)",190:"190-head(头)",191:"191-windmills(风车)",192:"192-giraffe(长颈鹿)",193:"193-iguana(鬣蜥)",194:"194-ice(冰)",195:"195-polar(极地)",196:"196-people(人)",197:"197-ceremony(仪式)",198:"198-church(教堂)",199:"199-castle(城堡)",200:"200-lizard(蜥蜴)",201:"201-rodent(啮齿动物)",202:"202-flowers(花)",203:"203-vehicle(车辆)",204:"204-beach(海滩)",205:"205-railroad(铁路)",206:"206-frost(霜)",207:"207-door(门)",208:"208-antelope(羚羊)",209:"209-house(房子)",210:"210-snow(雪)",211:"211-festival(节日)",212:"212-scotland(苏格兰)",213:"213-pots(锅)",214:"214-elk(麋鹿)",215:"215-cafe(咖啡馆)",216:"216-sun(太阳)",217:"217-prototype(原型)",218:"218-whales(鲸鱼)",219:"219-fox(狐狸)",220:"220-tracks(跑道)",221:"221-hut(小屋)",222:"222-harbor(港)",223:"223-plants(植物)",224:"224-cat(猫)",225:"225-glass(玻璃)",226:"226-hills(丘陵)",227:"227-barn(谷仓)",228:"228-nest(巢)",229:"229-cave(洞穴)",230:"230-town(小镇)",231:"231-antlers(鹿角)",232:"232-dock(码头)",233:"233-truck(卡车)",234:"234-swimmers(游泳者)",235:"235-garden(花园)",236:"236-wood(木)",237:"237-butterfly(蝴蝶)",238:"238-goat(山羊)",239:"239-stairs(楼梯)",240:"240-monks(僧侣)",241:"241-island(岛)",242:"242-frozen(冻)",243:"243-leaf(叶)",244:"244-museum(博物馆)",245:"245-cactus(仙人掌)",246:"246-ocean(海洋)",247:"247-hotel(酒店)",248:"248-girl(女孩)",249:"249-arch(拱)",250:"250-monument(纪念碑)",251:"251-farms(农场)",252:"252-park(公园)",253:"253-dance(跳舞)",254:"254-orchid(兰花)",255:"255-display(展示)",256:"256-athlete(运动员)",257:"257-plaza(广场)",258:"258-deer(鹿)",259:"259-sand(沙子)",260:"260-river(河)"}
IMAGE_PATH_PREFIX = "/home/niusilong/work/AI/training/image/"
IMAGE_SUFFIX = ".jpeg"
APPEND_PATH = False
ALL_LABELS = np.arange(start=1, stop=261, dtype=np.int32)
LABEL_MIN_COUNT = 30


class ImageLabel(object):
    def __init__(self, line, labels):
        self.line = line
        self.labels = labels
    def __str__(self):
        return self.line
def get_image_labels():
    original_image_lines = []
    with open(IMAGE_LABEL_FILE) as f:
        while True:
            line = f.readline().strip()
            if line == "":
                break
            line_array = line.split(" ")
            if len(line_array) == 1:
                original_image_lines.append(ImageLabel(line, []))
                continue
            labels = line_array[1].split(",")
            labels = np.asarray(labels, dtype=np.int32)
            original_image_lines.append(ImageLabel(line, labels))
    return original_image_lines
def get_label_count(lines):
    label_count = np.zeros(shape=[260], dtype=np.int32)
    for line in lines:
        line_array = line.strip().split(" ")
        if len(line_array) == 1:
            continue
        labels = np.asarray(line_array[1].split(","), dtype=np.int32)
        for label in labels:
            label_count[label-1] += 1
    return label_count
def print_label_count(label_count):
    sorted_index_array = label_count.argsort()[::-1]
    for i in range(len(sorted_index_array)):
        print("label:%s, count:%d" % (LABEL_ID_DICT[ALL_LABELS[sorted_index_array[i]]], label_count[sorted_index_array[i]]))

if __name__ == '__main__':
    with open(IMAGE_LABEL_FILE) as f:
        lines = f.readlines()
    label_count = get_label_count(lines)
    print_label_count(label_count)