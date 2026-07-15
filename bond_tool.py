import numpy as np
from scipy.optimize import newton

class FixedIncomeBond:
    """
    A class representing a Fixed Income Bond to calculate price, 
    Yield to Maturity (YTM), and interest rate risk metrics (Greeks).
    """
    def __init__(self, face_value: float, coupon_rate: float, years_to_maturity: float, payment_frequency: int = 2):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.years_to_maturity = years_to_maturity
        self.frequency = payment_frequency  # 2 = Semi-annual (US Treasury/Corporate standard)
        self.total_periods = int(years_to_maturity * payment_frequency)
        self.coupon_payment = (coupon_rate * face_value) / payment_frequency

    def price_from_yield(self, ytm: float) -> float:
        """Calculates bond price given an annualized Yield to Maturity (YTM)."""
        periodic_yield = ytm / self.frequency
        periods = np.arange(1, self.total_periods + 1)
        
        # Present value of coupons and principal
        pv_coupons = np.sum(self.coupon_payment / (1 + periodic_yield) ** periods)
        pv_principal = self.face_value / (1 + periodic_yield) ** self.total_periods
        
        return pv_coupons + pv_principal

    def calculate_ytm(self, market_price: float) -> float:
        """Solves for YTM given a market price using the Newton-Raphson numerical method."""
        def obj_func(ytm):
            return self.price_from_yield(ytm) - market_price
        
        try:
            return newton(obj_func, x0=self.coupon_rate)
        except RuntimeError:
            raise ValueError("YTM calculation did not converge. Verify inputs.")

    def calculate_risk_metrics(self, ytm: float) -> dict:
        """
        Calculates key interest rate risk metrics: 
        Macaulay Duration, Modified Duration, Convexity, and DV01.
        """
        periodic_yield = ytm / self.frequency
        periods = np.arange(1, self.total_periods + 1)
        
        cash_flows = np.full(self.total_periods, self.coupon_payment)
        cash_flows[-1] += self.face_value
        
        pv_cf = cash_flows / (1 + periodic_yield) ** periods
        bond_price = np.sum(pv_cf)
        
        # Macaulay Duration (Weighted average time to receive cash flows)
        mac_duration = np.sum(periods * pv_cf) / bond_price / self.frequency
        
        # Modified Duration (Price sensitivity to a 1% change in yield)
        mod_duration = mac_duration / (1 + periodic_yield)
        
        # Convexity (Curvature of the price-yield relationship)
        convexity = np.sum(periods * (periods + 1) * pv_cf) / (bond_price * (1 + periodic_yield) ** 2)
        convexity = convexity / (self.frequency ** 2)
        
        # DV01 (Dollar risk exposure per 1 basis point / 0.01% yield shift)
        dv01 = bond_price * mod_duration * 0.0001
        
        return {
            "Price": round(bond_price, 2),
            "Macaulay Duration": round(mac_duration, 3),
            "Modified Duration": round(mod_duration, 3),
            "Convexity": round(convexity, 3),
            "DV01": round(dv01, 4)
        }

# --- Demo Execution for Testing ---
if __name__ == "__main__":
    # Example: 10-Year Corporate Bond, 5% Coupon, Trading at $960
    bond = FixedIncomeBond(face_value=1000.0, coupon_rate=0.05, years_to_maturity=10.0)
    
    # Calculate Metrics
    implied_ytm = bond.calculate_ytm(960.0)
    metrics = bond.calculate_risk_metrics(implied_ytm)
    
    print("====================================================")
    print("      FIXED INCOME ANALYTICS REPORT                 ")
    print("====================================================")
    print(f"Market Price:        ${960.00:.2f}")
    print(f"Implied YTM:         {implied_ytm * 100:.3f}%")
    print(f"Macaulay Duration:   {metrics['Macaulay Duration']} Years")
    print(f"Modified Duration:   {metrics['Modified Duration']}%")
    print(f"DV01 (Risk per bp):  ${metrics['DV01']:.4f}")
    print(f"Convexity:           {metrics['Convexity']}")
    print("====================================================")
