import os
import random
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class ChallanPDFGenerator:
    def __init__(self):
        """Initializes the challan generator and ensures output directory exists."""
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "challans")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_challan(self, violation_id, vehicle_details, violation_type, fine_amount, nlp_message, evidence_image_path):
        """
        Generates a beautiful PDF traffic challan.
        
        Args:
            violation_id (int): DB entry id.
            vehicle_details (dict): Owner info containing name, phone, address, vehicle_number.
            violation_type (str): Type of violation.
            fine_amount (int): Fine amount in INR.
            nlp_message (str): NLP generated notification message.
            evidence_image_path (str): Filepath to the violation screenshot.
        """
        # Create unique filename
        filename = f"challan_{violation_id}_{vehicle_details['vehicle_number']}.pdf"
        pdf_path = os.path.join(self.output_dir, filename)
        
        # Setup document template with 0.5-inch margins
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        story = []
        
        # Styles setup
        styles = getSampleStyleSheet()
        
        # Custom Title / Header styles
        header_title_style = ParagraphStyle(
            name='HeaderTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=colors.white,
            alignment=1, # Centered
            spaceAfter=6
        )
        header_subtitle_style = ParagraphStyle(
            name='HeaderSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=colors.lightgrey,
            alignment=1,
            spaceAfter=2
        )
        
        section_heading_style = ParagraphStyle(
            name='SectionHeading',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor('#002B49'), # Deep navy blue
            spaceBefore=10,
            spaceAfter=6
        )
        
        body_style = ParagraphStyle(
            name='BodyStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10.5,
            leading=14,
            textColor=colors.HexColor('#333333')
        )
        
        callout_style = ParagraphStyle(
            name='CalloutStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=11,
            leading=16,
            textColor=colors.HexColor('#1E293B'),
            alignment=4 # Justify
        )
        
        # 1. Header Banner Box
        # We represent this as a 1x1 table with a solid background color
        header_content = [
            Paragraph("DEPARTMENT OF TRAFFIC POLICE", header_subtitle_style),
            Paragraph("OFFICIAL TRAFFIC E-CHALLAN", header_title_style),
            Paragraph(f"Challan Reference Number: TPC-{random.randint(100000, 999999)}-{violation_id:04d}", header_subtitle_style),
            Paragraph(f"Date of Incident: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}", header_subtitle_style)
        ]
        
        header_table = Table([[header_content]], colWidths=[540])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#002B49')), # Dark blue
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 15),
            ('BOTTOMPADDING', (0,0), (-1,-1), 15),
            ('LEFTPADDING', (0,0), (-1,-1), 15),
            ('RIGHTPADDING', (0,0), (-1,-1), 15),
            ('BOTTOMMARGIN', (0,0), (-1,-1), 10),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 10))
        
        # 2. Section: Owner details
        story.append(Paragraph("I. Offender & Vehicle Details", section_heading_style))
        owner_data = [
            [Paragraph("<b>Vehicle Number</b>", body_style), Paragraph(vehicle_details['vehicle_number'], body_style)],
            [Paragraph("<b>Owner Name</b>", body_style), Paragraph(vehicle_details['owner_name'], body_style)],
            [Paragraph("<b>Mobile Number</b>", body_style), Paragraph(vehicle_details['phone'], body_style)],
            [Paragraph("<b>Registered Address</b>", body_style), Paragraph(vehicle_details['address'], body_style)]
        ]
        owner_table = Table(owner_data, colWidths=[150, 390])
        owner_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F1F5F9')), # Off-white/grey header col
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(owner_table)
        story.append(Spacer(1, 12))
        
        # 3. Section: Violation Details
        story.append(Paragraph("II. Violation Information & Penalty", section_heading_style))
        violation_data = [
            [Paragraph("<b>Violation Description</b>", body_style), Paragraph(f"<font color='red'><b>{violation_type.upper()}</b></font>", body_style)],
            [Paragraph("<b>Fine Amount</b>", body_style), Paragraph(f"<b>₹{fine_amount}</b> (Rupees One Thousand Only)", body_style)],
            [Paragraph("<b>Payment Status</b>", body_style), Paragraph("<font color='orange'><b>PENDING</b></font>", body_style)]
        ]
        violation_table = Table(violation_data, colWidths=[150, 390])
        violation_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F1F5F9')),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(violation_table)
        story.append(Spacer(1, 12))
        
        # 4. Section: NLP Notification Message
        story.append(Paragraph("III. AI NLP-Generated Notice", section_heading_style))
        nlp_box = Table([[Paragraph(nlp_message, callout_style)]], colWidths=[540])
        nlp_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EFF6FF')), # Light blue highlight
            ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor('#3B82F6')), # Blue border
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ]))
        story.append(nlp_box)
        story.append(Spacer(1, 12))
        
        # 5. Section: Evidence Image
        story.append(Paragraph("IV. Photographic Evidence", section_heading_style))
        if evidence_image_path and os.path.exists(evidence_image_path):
            try:
                # Add image, resizing to fit neatly on page
                # Max width is 300, keeping aspect ratio
                evidence_img = Image(evidence_image_path, width=320, height=180)
                evidence_img.hAlign = 'CENTER'
                story.append(evidence_img)
            except Exception as e:
                story.append(Paragraph(f"<i>Error loading evidence image: {e}</i>", body_style))
        else:
            story.append(Paragraph("<i>No photographic evidence image attached.</i>", body_style))
            
        story.append(Spacer(1, 15))
        
        # 6. Footer section (QR code mock and terms)
        terms_style = ParagraphStyle(
            name='TermsStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=10,
            textColor=colors.HexColor('#64748B'),
            alignment=1 # Centered
        )
        story.append(Paragraph("This is an electronically generated document. No physical signature is required under Section 4 of IT Act 2000.", terms_style))
        story.append(Paragraph("Please visit the official Traffic Department web portal to complete your fine payment within 15 days.", terms_style))
        
        # Build Document
        doc.build(story)
        return pdf_path

if __name__ == "__main__":
    generator = ChallanPDFGenerator()
    dummy_owner = {
        "vehicle_number": "MH12AB1234",
        "owner_name": "Rahul Sharma",
        "phone": "+91 98765 43210",
        "address": "Flat 402, Sunshine Apartments, Pune, Maharashtra - 411001"
    }
    nlp_msg = "Dear Rahul Sharma, your vehicle MH12AB1234 was detected violating traffic rules due to riding without a helmet on 2026-07-12. A fine of ₹1000 has been generated."
    
    # Run test without image
    pdf = generator.generate_challan(1, dummy_owner, "Riding without Helmet", 1000, nlp_msg, None)
    print("Test PDF challan generated at:", pdf)
