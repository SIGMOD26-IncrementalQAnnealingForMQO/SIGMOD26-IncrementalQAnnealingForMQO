package mqo_chimera.testcases;

import java.io.Serializable;

/**
 * Represents an interference between two query plans of two different queries in case both of
 * them are selected for execution. This interference can be positive, meaning that cost is
 * saved if both plans can share intermediate results for instance, or it can be negative
 * meaning that executing both plans causes for instance some synchronization overhead.
 * 
 * @author immanueltrummer
 *
 */
public class PlanCoupling implements Serializable {
	/**
	 * Used to verify class version.
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * Query to which the first plan applies
	 */
	public final int query1;
	/**
	 * First plan
	 */
	public final int plan1;
	/**
	 * Query to which the second plan applies
	 */	
	public final int query2;
	/**
	 * Second plan
	 */
	public final int plan2;

	public final double costSaving;
	
	public PlanCoupling(int query1, int plan1, int query2, int plan2) {
		this.query1 = query1;
		this.plan1 = plan1;
		this.query2 = query2;
		this.plan2 = plan2;
		this.costSaving = 0;
	}

	public PlanCoupling(int query1, int plan1, int query2, int plan2, double costSaving) {
		this.query1 = query1;
		this.plan1 = plan1;
		this.query2 = query2;
		this.plan2 = plan2;
		this.costSaving = costSaving;
	}
	/**
	 * Two couplings are equivalent if they connect the same plans of the same queries.
	 */
	@Override
	public boolean equals(Object otherObject) {
		PlanCoupling otherCoupling = (PlanCoupling)otherObject;
		return query1 == otherCoupling.query1 && query2 == otherCoupling.query2 && 
				plan1 == otherCoupling.plan1 && plan2 == otherCoupling.plan2 ||
				query1 == otherCoupling.query2 && query2 == otherCoupling.query1 && 
				plan1 == otherCoupling.plan2 && plan2 == otherCoupling.plan1;
	}
	@Override
	/*public int hashCode() {
		return query1 + query2 + plan1 + plan2; 
	}*/

	public int hashCode() {
		String hashString = "";
		if (query1 < query2) {
			hashString = "Q" + query1 + "P" + plan1 + "<->" + "Q" + query2 + "P" + plan2;
		} else {
			hashString = "Q" + query2 + "P" + plan2 + "<->" + "Q" + query1 + "P" + plan1;
		}

		return hashString.hashCode();
	}

	@Override
	public String toString() {
		return "Q" + query1 + "P" + plan1 + "<->" + "Q" + query2 + "P" + plan2;
	}
}
