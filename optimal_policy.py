"""
Optimal Bakım Politikası Hesaplama Modülü
Bu modül, çok bileşenli sistemler için optimal bakım politikası hesaplamalarını yapar.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

def StateIndex(StateDescV, Dim, SetSize):
    """
    State tanımlama vektörünü doğrusal indekse dönüştürür.
    
    Arguments:
        StateDescV: State tanımlama vektörü
        Dim: Boyut (bileşen sayısı)
        SetSize: Küme boyutu (K+1)
    
    Returns:
        int: Doğrusal indeks
    """
    Index = 0
    for i in range(Dim):
        Index += StateDescV[i]*((SetSize)**i)
    return int(Index)

def StateDesc(Ind, Dim, SetSize):
    """
    Doğrusal indeksten state tanımlama vektörünü hesaplar.
    
    Arguments:
        Ind: Doğrusal indeks
        Dim: Boyut (bileşen sayısı)
        SetSize: Küme boyutu (K+1)
    
    Returns:
        numpy.array: State tanımlama vektörü
    """
    StateDescV = np.zeros(Dim, dtype=int)
    for i in range(Dim):
        StateDescV[i] = np.remainder(Ind, SetSize)
        Ind = np.floor_divide(Ind, SetSize)
    return StateDescV

def PreProcessing(alpha, K, C, Eps):
    """
    Optimal politika hesaplamadan önce gereken ön işlemleri yapar.
    
    Arguments:
        alpha: Bozulmama olasılığı
        K: Maksimum bozulma seviyesi 
        C: Bileşen sayısı
        Eps: Hassasiyet eşiği
    
    Returns:
        tuple: (NumberNonGreenV, DistributionV, ObsTransM, t+1)
    """
    DetSSize = K+1
    ProbReachingT = 1
    
    StateSpacesSize = DetSSize**C  # Ortak dağılım için durum uzayı boyutu
    TransitionM = np.zeros([StateSpacesSize, StateSpacesSize])
    DistributionV = np.zeros([1, StateSpacesSize])
    CondDistV = np.zeros(StateSpacesSize)
    ObsTransM = np.zeros([1, 2])  # t sayacındaki Sarı durumlardan Sarı veya Kırmızı durumlara geçiş olasılıkları
    
    NotRedMaskV = np.zeros(StateSpacesSize, dtype=int)
    NumberNonGreenV = np.zeros(StateSpacesSize, dtype=int)
    StateDescVector = np.zeros(C)
    
    # Kırmızı durumların listesini ve kırmızı olmayan durumların maskeleme vektörünü oluştur
    RedList = []
    for Ind in range(StateSpacesSize):
        StateV = StateDesc(Ind, C, DetSSize)
        
        # Kırmızı durumları tespit et
        RedState = 0
        for i in range(C):
            if StateV[i] == K:
                RedState = 1
                break
        
        if RedState == 0:
            NotRedMaskV[Ind] = 1
        else:
            RedList.append(Ind)
        
        # Yeşil olmayan bileşenleri say
        NumNonGreen = 0
        for i in range(C):
            if StateV[i] != 0:
                NumNonGreen += 1
        
        NumberNonGreenV[Ind] = NumNonGreen
    
    # Geçiş matrisini oluştur
    for FromInd in range(StateSpacesSize):
        FromStateV = StateDesc(FromInd, C, DetSSize)
        
        if FromInd in RedList:
            TransitionM[FromInd, FromInd] = 1
        else:
            for IncInd in range(2**C):
                IncV = StateDesc(IncInd, C, 2)
                
                ToStateV = FromStateV + IncV
                ToStateInd = StateIndex(ToStateV, C, DetSSize)
                TransitionM[FromInd, ToStateInd] = (1-alpha)**(sum(IncV))*(alpha**(C-sum(IncV)))
    
    t = 0
    DistributionV[t, :] = TransitionM[0, :]
    CondDistV = 1/(1-DistributionV[0, 0])*DistributionV[t, :]
    CondDistV[0] = 0
    
    ObsTransM[0, :] = [1, 0]
    
    DistributionV[t, :] = CondDistV
    
    # Hassasiyet eşiği karşılanana kadar döngü
    while ProbReachingT > Eps:
        t += 1
        DistributionV = np.append(DistributionV, [np.matmul(CondDistV, TransitionM)], axis=0)
        CondDistV = np.multiply(DistributionV[t, :], NotRedMaskV)
        SumNotRed = np.sum(CondDistV)
        ObsTransM = np.append(ObsTransM, [[SumNotRed, 1-SumNotRed]], axis=0)
        ProbReachingT *= ObsTransM[t, 0]
        CondDistV = 1/SumNotRed*CondDistV
    
    return NumberNonGreenV, DistributionV, ObsTransM, t+1

def ReliabilityLP(Code, K, C, U, alpha, cp, cc, cr, cu, co, ct, NumberNonGreenV, DistributionV, ObsTransM):
    """
    Bakım politikası optimizasyonu için doğrusal programlama modelini çözer.
    
    Arguments:
        Code: Politika kodu (0 için optimal politika)
        K: Maksimum bozulma seviyesi
        C: Bileşen sayısı
        U: Zaman kesinti noktası
        alpha: Bozulmama olasılığı
        cp: Önleyici bakım maliyeti
        cc: Düzeltici bakım maliyeti
        cr: Yedek parça değiştirme maliyeti
        cu: Eksik parça maliyeti
        co: Fazla parça maliyeti
        ct: Parça transfer maliyeti
        NumberNonGreenV: Yeşil olmayan bileşen sayısı vektörü
        DistributionV: Dağılım vektörü
        ObsTransM: Geçiş matrisi
    
    Returns:
        tuple: (SolutionMat, ObjValue)
    """
    DetSSize = K+1
    CounterStuckP = alpha**C
    StateSpacesSize = DetSSize**C  
    
    PeriodCosts = np.zeros((2, U+1, C+1))
    PeriodCosts[0, :, 1:C+1] = cp
    PeriodCosts[1, :, :] = cc
    
    for s in range(2):
        for n in range(0, U+1):  
            for a in range(1, C+1):
                PeriodCosts[s, n, a] += ct*a
                
                if n == 0:
                    PeriodCosts[s, n, a] += (cr*NumberNonGreenV[0] + 
                                           cu*max(NumberNonGreenV[0]-a, 0) + 
                                           co*max(-NumberNonGreenV[0]+a, 0))
                else:
                    for Ind in range(StateSpacesSize):
                        PeriodCosts[s, n, a] += (cr*NumberNonGreenV[Ind] + 
                                               cu*max(NumberNonGreenV[Ind]-a, 0) + 
                                               co*max(-NumberNonGreenV[Ind]+a, 0))*DistributionV[n-1, Ind]
    
    # Model oluştur
    lp = gp.Model("ReliabilityMDP")
    lp.setParam("FeasibilityTol", 1e-9)
    lp.setParam("NumericFocus", 2)
    
    # Optimize edilecek değişkenler
    P = lp.addMVar(shape=(2, U+1, C+1), name="P", obj=PeriodCosts)
    
    # Kısıt 0: Durum 0'da müdahale yok
    lp.addConstr(
        gp.quicksum(P[0, 0, 1:C+1]) == 0
    )
    
    # Kısıt 1
    lp.addConstr(
        gp.quicksum(P[0, 0, :]) - 
        gp.quicksum(P[s, n, a] 
            for s in range(2)
            for n in range(U+1)
            for a in range(1, C+1)
        ) - 
        CounterStuckP*P[0, 0, 0] == 0
    )
    
    # Kısıt 2
    lp.addConstr(
        gp.quicksum(P[0, 1, :]) - 
        (1-CounterStuckP)*P[0, 0, 0] == 0
    )
    
    # Kısıt 3 ve 4
    for n in range(2, U+1):
        lp.addConstr(
            gp.quicksum(P[0, n, :]) - 
            ObsTransM[n-1, 0]*P[0, n-1, 0] == 0
        )
        
        lp.addConstr(
            gp.quicksum(P[1, n, :]) - 
            ObsTransM[n-1, 1]*P[0, n-1, 0] == 0
        )
    
    # Kısıt 5: Toplam olasılık 1
    lp.addConstr(
        P.sum() == 1
    )
    
    # Kısıt: Kırmızıda müdahale etmek zorunlu
    for n in range(U+1):
        lp.addConstr(
            P[1, n, 0] == 0
        )
    
    # Kısıt: K'dan önce kırmızı duruma ulaşılamaz
    for n in range(K):
        for a in range(C+1):
            lp.addConstr(
                P[1, n, a] == 0
            )
    
    # Kısıt: Sayaç kesinti noktasında müdahale etmek zorunlu
    lp.addConstr(
        P[0, U, 0] == 0
    )
    
    # Farklı politika kodlarına göre ek kısıtlar
    if Code == 1:  # K-1'de optimal sayıda parça ile müdahale et
        for n in range(K-1):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
        lp.addConstr(P[0, K-1, 0] == 0)
    
    elif Code == 2:  # K-1'de sadece 1 parça ile müdahale et
        for n in range(K-1):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
        lp.addConstr(P[0, K-1, 0] == 0)
        for a in range(2, C+1):
            lp.addConstr(P[0, K-1, a] == 0)
    
    elif Code == 3:  # K-1'de C parça ile müdahale et
        for n in range(K-1):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
        lp.addConstr(P[0, K-1, 0] == 0)
        for a in range(1, C):
            lp.addConstr(P[0, K-1, a] == 0)
    
    elif Code == 4:  # Sadece kırmızıda optimal sayıda parça ile müdahale et
        for n in range(U):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
    
    elif Code == 5:  # Sadece kırmızıda 1 parça ile müdahale et
        for n in range(U):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
        for n in range(U+1):
            for a in range(2, C+1):
                lp.addConstr(P[1, n, a] == 0)
    
    elif Code == 6:  # Sadece kırmızıda C parça ile müdahale et
        for n in range(U):
            for a in range(1, C+1):
                lp.addConstr(P[0, n, a] == 0)
        for n in range(U+1):
            for a in range(1, C):
                lp.addConstr(P[1, n, a] == 0)
    
    # Modeli çöz
    lp.optimize()
    
    # Çözümü al
    SolutionMat = P.X
    ObjValue = lp.getObjective().getValue()
    
    return SolutionMat, ObjValue

def calculate_optimal_policy(params):
    """
    Verilen parametrelere göre optimal bakım politikasını hesaplar.
    
    Arguments:
        params: Parametre sözlüğü (C, K, alpha, cp, cc, cr, cu, co, ct)
    
    Returns:
        dict: Optimal politika bilgileri
    """
    try:
        # Parametreleri ayıkla
        C = params['C']
        K = params['K']
        alpha = params['alpha']
        cp = params['c1']
        cc = params['c2']
        cr = params['cr']
        cu = params['cs']
        co = params['ce']
        ct = params['ct']
        
        # Hassasiyet eşiği
        Eps = 0.01
        
        # Ön işleme
        NumberNonGreenV, DistributionV, ObsTransM, U = PreProcessing(alpha, K, C, Eps)
        
        # Optimal politikayı hesapla (Code=0 optimal politika için)
        SolutionMat, ObjValue = ReliabilityLP(0, K, C, U, alpha, cp, cc, cr, cu, co, ct, 
                                            NumberNonGreenV, DistributionV, ObsTransM)
        
        # Optimal politika bilgilerini çıkart
        policy_info = extract_policy_info(SolutionMat, C, U, ObjValue, NumberNonGreenV, DistributionV)
        
        return {
            'success': True,
            'policy': policy_info,
            'objective_value': ObjValue,
            'solution_matrix': SolutionMat.tolist()  # Numpy dizisini normal listeye çevir
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def extract_policy_info(SolutionMat, C, U, objective, NumberNonGreenV, DistributionV):
    """
    Çözüm matrisinden politika bilgilerini çıkarır.
    
    Arguments:
        SolutionMat: Çözüm matrisi
        C: Bileşen sayısı
        U: Zaman kesinti noktası
        objective: Amaç fonksiyonu değeri
        NumberNonGreenV: Yeşil olmayan bileşen sayısı vektörü
        DistributionV: Dağılım vektörü
    
    Returns:
        dict: Politika bilgileri
    """
    # Hangi durumda müdahale edildiğini bul
    yellow_interventions = []
    red_interventions = []
    
    # Sarı durumlar için
    for n in range(U+1):
        for a in range(1, C+1):
            if SolutionMat[0, n, a] > 0.001:  # Küçük sayısal hatalar için tolerans
                yellow_interventions.append({
                    'counter': int(n),
                    'components': int(a),
                    'probability': float(SolutionMat[0, n, a])
                })
    
    # Kırmızı durumlar için
    for n in range(U+1):
        for a in range(1, C+1):
            if SolutionMat[1, n, a] > 0.001:
                red_interventions.append({
                    'counter': int(n),
                    'components': int(a),
                    'probability': float(SolutionMat[1, n, a])
                })
    
    # Sistemin kırmızı duruma düşme olasılığı
    down_probability = float(np.sum(SolutionMat[1, :, :]))
    
    # Sistemin önleyici bakım yapılma olasılığı
    preventive_probability = float(np.sum(SolutionMat[0, :, 1:C+1]))
    
    # Optimal politikanın özeti
    if len(yellow_interventions) == 1:
        policy_description = f"Optimal bakım politikası: Sarı sinyal sayacı {yellow_interventions[0]['counter']} değerine ulaştığında {yellow_interventions[0]['components']} bileşen ile önleyici bakım yap. Kırmızı sinyalde düzeltici bakım yap."
    elif len(yellow_interventions) > 1:
        policy_description = "Karma bakım politikası: Birden fazla senaryoda önleyici bakım yapılmalı."
    elif len(red_interventions) > 0:
        policy_description = "Sadece düzeltici bakım yapılmalı (kırmızı sinyal durumunda)"
    else:
        policy_description = "Geçerli bir bakım politikası bulunamadı."
    
    return {
        'objective_value': float(objective),
        'yellow_interventions': yellow_interventions,
        'red_interventions': red_interventions,
        'down_probability': down_probability,
        'preventive_probability': preventive_probability,
        'policy_description': policy_description
    }