<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:ad="http://www.lexisnexis.com/namespace/sslrp/ad"
    xmlns:tr="http://www.lexisnexis.com/namespace/sslrp/tr"
    xmlns:ps="http://www.lexisnexis.com/namespace/sslrp/ps"
    xmlns:form="http://www.lexisnexis.com/namespace/sslrp/form"
    xmlns:fn="http://www.lexisnexis.com/namespace/sslrp/fn"
    xmlns:se="http://www.lexisnexis.com/namespace/sslrp/se"
    xmlns:di="http://www.lexisnexis.com/namespace/sslrp/di"
    xmlns:core="http://www.lexisnexis.com/namespace/sslrp/core"
    xmlns:em="http://www.lexisnexis.com/namespace/sslrp/em"
    xmlns:fm="http://www.lexisnexis.com/namespace/sslrp/fm"
    xmlns:pnfo="http://www.lexisnexis.com/namespace/sslrp/pnfo"
    xmlns:lnci="http://www.lexisnexis.com/namespace/common/lnci"
    xmlns:glph="http://www.lexisnexis.com/namespace/sslrp/glph"
    xmlns:pu="http://www.lexisnexis.com/namespace/sslrp/pu"
    xmlns:su="http://www.lexisnexis.com/namespace/sslrp/su"
    xmlns:nl="http://www.lexisnexis.com/namespace/sslrp/nl"
    xmlns:ls="http://www.lexisnexis.com/namespace/sslrp/ls"
    xmlns:nc="http://www.lexisnexis.com/namespace/sslrp/nc" exclude-result-prefixes="nc ls nl su pu glph lnci pnfo fm em core di se fn form ps tr">

    <xsl:output method="html" encoding="us-ascii" />

    <!-- Copy anything not caught by any of the specific styles -->
    <xsl:template match="@*|node()" mode="verbatim">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()" mode="verbatim"/>
        </xsl:copy>
    </xsl:template>

    <xsl:template match="ad:adjudication">
        <html>
            <head>
                <style>
            @page {
            size: letter portrait;
            margin: 0.75in;
            }
     	    @media print and (color) {
            * {
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
            }
            .core-juris-name {float: left; overflow: auto;}
            .ad-decisiondate {float: right;}
	        .ad-typeofcase h4 {background-color: #FFF000;}
            .ad-regarding {background-color: #F0FFF0;}
            .ad-filenum {background-color: #FFFAFA;}
            .core-juris-name {background-color: #F5FFFA;}
            .ad-decisiondate {background-color: #FFF0F5;}
            .ad-typeofcase {color: blue;}
            .core-flag {background-color: #FFFDD0;}
            .core-flag p {background-color: #FFFDD0 !important; margin: 0 !important; border-style: none !important;}
            .core-flag-subject {margin: 0;}
            p {background-color: #FFFFFF;}
            small {background-color: #F0FFF0;}
            h4 {background-color: #FFF0FF;}
            ul p {background-color: #FFF0F0;}
            i {background-color: #FAEBD7;}
            b {background-color: #F0F0FF;}
                </style>
            </head>
            <body>
                <xsl:apply-templates />
                <hr/>
            </body>
        </html>
    </xsl:template>

    <xsl:template match="ad:regarding">
        <div class="ad-regarding">
            <xsl:apply-templates select=".//text()"/>
        </div>
    </xsl:template>

    <xsl:template match="ad:filenum">
        <div class="ad-filenum">
            <xsl:apply-templates select=".//text()"/>
        </div>
    </xsl:template>

    <xsl:template match="core:juris-name">
        <div class="core-juris-name">
            <xsl:apply-templates select=".//text()"/>
        </div>
    </xsl:template>
    <xsl:template match="ad:decisiondate">
        <div class="ad-decisiondate">
            <xsl:apply-templates select=".//text()"/>
        </div>
    </xsl:template>
    <xsl:template match="ad:typeofcase">
        <div class="ad-typeofcase">
            <xsl:apply-templates/>
        </div>
    </xsl:template>

    <!-- Lists -->
    <xsl:template match="core:list">
        <ul>
            <xsl:apply-templates />
        </ul>
    </xsl:template>

    <xsl:template match="core:listitem">
        <li>
            <xsl:apply-templates />
        </li>
    </xsl:template>
    <!-- Flags -->
    <xsl:template match="core:flag">
        <div class="core-flag">
            <xsl:apply-templates />
        </div>
    </xsl:template>

    <!-- Flags - Subject -->
    <xsl:template match="core:flag-subject">
        <p class="core-flag-subject">
            <xsl:apply-templates />
        </p>
    </xsl:template>

    <!-- Flags - Message -->
    <xsl:template match="core:message">
        <div class="core-message">
            <xsl:apply-templates />
        </div>
    </xsl:template>

    <!-- Generic Head -->
    <xsl:template match="core:generic-hd">
        <h4>
            <xsl:apply-templates />
        </h4>
    </xsl:template>

    <!-- Paras. -->
    <xsl:template match="core:para | core:flag.para">
        <p>
            <xsl:apply-templates />
        </p>
    </xsl:template>

    <!-- Blockquotes. Use the HTML blockquote element. -->
    <xsl:template match="core:blockquote">
        <blockquote>
            <xsl:apply-templates />
        </blockquote>
    </xsl:template>

    <!-- Footnotes. Shrink the text and put square brackets around it. -->
    <xsl:template match="fn:footnote">
        <small>
            <xsl:text>[n</xsl:text>
            <xsl:value-of select="@fr"/>
            <xsl:text>: </xsl:text>
            <xsl:apply-templates />
            <xsl:text>]</xsl:text>
        </small>
    </xsl:template>

    <xsl:template match="fn:footnote" mode="normal">
        <small>
            <xsl:text>[n</xsl:text>
            <xsl:value-of select="@fr"/>
            <xsl:text>: </xsl:text>
            <xsl:apply-templates />
            <xsl:text>]</xsl:text>
        </small>
    </xsl:template>


    <xsl:template match="fn:para">
        <xsl:apply-templates />
        <xsl:text></xsl:text>
    </xsl:template>
    <!-- Text formatting. -->
    <xsl:template match="core:emph[@typestyle='it']">
        <i>
            <xsl:apply-templates />
        </i>
    </xsl:template>

    <xsl:template match="core:emph[@typestyle='bf']">
        <b>
            <xsl:apply-templates />
        </b>
    </xsl:template>

    <xsl:template match="core:emph[@typestyle='ib']">
        <i>
            <b>
                <xsl:apply-templates />
            </b>
        </i>
    </xsl:template>

    <xsl:template match="core:emph[@typestyle='un']">
        <u>
            <xsl:apply-templates />
        </u>
    </xsl:template>

    <xsl:template match="core:emph[@typestyle='smcaps']">
        <span style="font-variant: small-caps;">
            <xsl:apply-templates/>
        </span>
    </xsl:template>

    <xsl:template match="core:nl">
        <br/>
    </xsl:template>

    <!-- Tables. At least warn when they're present. -->
    <xsl:template match="table">
        <xsl:message terminate="no">
            <xsl:text>*** table ***</xsl:text>
        </xsl:message>
        <table>
            <xsl:apply-templates/>
        </table>
    </xsl:template>

    <xsl:template match="tgroup">
        <colgroup>
            <xsl:apply-templates select="colspec"/>
        </colgroup>
        <xsl:apply-templates/>
    </xsl:template>

    <xsl:template match="colspec">
        <col>
            <xsl:attribute name="width">
                <xsl:value-of select=" @colwidth "/>
            </xsl:attribute>
        </col>
    </xsl:template>

    <xsl:template match="row">
        <tr>
            <xsl:copy-of select="@valign"/>
            <xsl:apply-templates />
        </tr>
    </xsl:template>

    <xsl:template match="entry">
        <td>
            <xsl:copy-of select="@align"/>
            <xsl:copy-of select="@valign"/>
            <xsl:apply-templates />
        </td>
    </xsl:template>

    <!-- Copy illo markup verbatim. -->
    <xsl:template match="core:figure">
        <xmp>
            <xsl:copy>
                <xsl:apply-templates select="@*" mode="verbatim"/>
                <xsl:call-template name="newline"/>
                <xsl:apply-templates select="node()" mode="verbatim"/>
                <xsl:call-template name="newline"/>
            </xsl:copy>
        </xmp>
    </xsl:template>


    <xsl:template name="newline">
        <xsl:text>
        </xsl:text>
    </xsl:template>


    <!-- Cites -->
    <xsl:template match="lnci:cite">
        <xsl:apply-templates select=".//lnci:content/node()"/>
    </xsl:template>

</xsl:stylesheet>
