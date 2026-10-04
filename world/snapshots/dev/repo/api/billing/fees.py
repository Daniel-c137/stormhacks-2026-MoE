"""Our success fee, charged only after a cancellation is confirmed (#20)."""

from decimal import ROUND_HALF_UP, Decimal

# Share of the first year's savings we charge.
FEE_RATE = Decimal("0.30")


def first_year_savings(monthly_price: Decimal) -> Decimal:
    return monthly_price * 12


def success_fee(monthly_price: Decimal, cancellation_confirmed: bool) -> Decimal:
    """The fee for one cancelled subscription; nothing until the provider confirms it."""
    if not cancellation_confirmed:
        return Decimal("0")
    fee = first_year_savings(monthly_price) * FEE_RATE
    return fee.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
