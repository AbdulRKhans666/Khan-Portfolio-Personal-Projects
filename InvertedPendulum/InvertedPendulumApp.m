classdef InvertedPendulumApp < matlab.apps.AppBase

% ================================================================
%  INVERTED PENDULUM — LQR Interactive GUI
%  ~Abdul K.
% ================================================================
%  REQUIRES: Control System Toolbox
%
%  ANGLE CONVENTION:
%    theta = pi  ->  pendulum UPRIGHT  (target)
%    theta = 0   ->  pendulum hanging DOWN
% ================================================================

    properties (Access = public)
        UIFigure        matlab.ui.Figure
        PanelLeft       matlab.ui.container.Panel
        EditPendMass    matlab.ui.control.NumericEditField
        EditCartMass    matlab.ui.control.NumericEditField
        EditArmLen      matlab.ui.control.NumericEditField
        EditGravity     matlab.ui.control.NumericEditField
        EditFriction    matlab.ui.control.NumericEditField
        EditQ1          matlab.ui.control.NumericEditField
        EditQ2          matlab.ui.control.NumericEditField
        EditQ3          matlab.ui.control.NumericEditField
        EditQ4          matlab.ui.control.NumericEditField
        EditR           matlab.ui.control.NumericEditField
        EditTx          matlab.ui.control.NumericEditField
        EditTtheta      matlab.ui.control.NumericEditField
        EditIx          matlab.ui.control.NumericEditField
        EditIv          matlab.ui.control.NumericEditField
        EditItheta      matlab.ui.control.NumericEditField
        EditIomega      matlab.ui.control.NumericEditField
        EditTend        matlab.ui.control.NumericEditField
        EditDt          matlab.ui.control.NumericEditField
        BtnRun          matlab.ui.control.Button
        BtnStop         matlab.ui.control.Button
        LblStatus       matlab.ui.control.Label
        LblGains        matlab.ui.control.Label
        PanelRight      matlab.ui.container.Panel
        AxAnim          matlab.ui.control.UIAxes
        AxX             matlab.ui.control.UIAxes
        AxV             matlab.ui.control.UIAxes
        AxTheta         matlab.ui.control.UIAxes
        AxOmega         matlab.ui.control.UIAxes
    end

    properties (Access = private)
        StopFlag logical = false
    end

    % ================================================================
    %  SIMULATION LOGIC
    % ================================================================
    methods (Access = private)

        function runSimulation(app)
            app.StopFlag = false;
            app.BtnRun.Enable = 'off';
            app.setStatus('Building model...', [1.0 0.75 0.2]);
            drawnow;

            % Read inputs
            pm  = app.EditPendMass.Value;
            cm  = app.EditCartMass.Value;
            al  = app.EditArmLen.Value;
            gv  = app.EditGravity.Value;
            dv  = app.EditFriction.Value;
            Q   = diag([app.EditQ1.Value, app.EditQ2.Value, ...
                        app.EditQ3.Value, app.EditQ4.Value]);
            Rv  = app.EditR.Value;
            wr  = [app.EditTx.Value; 0; app.EditTtheta.Value; 0];
            x0  = [app.EditIx.Value; app.EditIv.Value; ...
                   app.EditItheta.Value; app.EditIomega.Value];
            tf  = app.EditTend.Value;
            dt  = app.EditDt.Value;
            t   = (0:dt:tf)';
            N   = length(t);

            % State-space model
            A = [0,  1,              0,                    0;
                 0, -dv/cm,          pm*gv/cm,             0;
                 0,  0,              0,                    1;
                 0, -dv/(cm*al),    -(pm+cm)*gv/(cm*al),   0];
            B = [0; 1/cm; 0; 1/(cm*al)];

            % Controllability check
            if rank(ctrb(A,B)) < 4
                app.setStatus('ERROR: System not controllable!', [1 0.3 0.3]);
                app.BtnRun.Enable = 'on';
                return
            end

            % LQR design
            try
                K = lqr(A, B, Q, Rv);
            catch ex
                app.setStatus(['LQR failed: ' ex.message], [1 0.3 0.3]);
                app.BtnRun.Enable = 'on';
                return
            end
            app.LblGains.Text = sprintf('K = [%.2f, %.2f, %.2f, %.2f]', K);

            % Integrate ODE
            app.setStatus('Integrating ODE...', [1.0 0.75 0.2]);
            drawnow;
            opts = odeset('RelTol',1e-6,'AbsTol',1e-8);
            [t_sol, sol] = ode45(@(t,x) pendcart_ode(x,K,wr,pm,cm,al,gv,dv), ...
                                  t, x0, opts);

            % Setup animation axes
            app.setupAnimAxes(wr, al);

            % Setup state plot axes
            stAxes  = {app.AxX, app.AxV, app.AxTheta, app.AxOmega};
            sTitles = {'x  (m)', 'v  (m/s)', '\theta  (rad)', '\omega  (rad/s)'};
            sCols   = {[0.30 0.70 1.0],[0.25 0.95 0.60],[1.0 0.80 0.25],[1.0 0.38 0.48]};
            refVals = [wr(1), 0, wr(3), 0];
            hLines  = cell(4,1);
            for i = 1:4
                cla(stAxes{i}); hold(stAxes{i},'on'); grid(stAxes{i},'on');
                xlim(stAxes{i},[0 tf]);
                title(stAxes{i}, sTitles{i},'Color',[0.65 0.80 0.95],...
                      'FontSize',9,'FontName','Courier New');
                xlabel(stAxes{i},'Time (s)','Color',[0.42 0.52 0.62],'FontSize',8);
                yline(stAxes{i}, refVals(i),'--','Color',[0.42 0.46 0.52],'LineWidth',1.0);
                hLines{i} = plot(stAxes{i}, nan, nan, '-', ...
                                 'Color', sCols{i}, 'LineWidth', 2.0);
            end

            % Build animation graphics
            ax = app.AxAnim;
            hw_l = scatter(ax, 0, 0.25, 500, [0.22 0.26 0.32],'filled');
            hw_r = scatter(ax, 0, 0.25, 500, [0.22 0.26 0.32],'filled');
            hbdy = plot(ax, [0 0],[0.65 0.65],'-','Color',[0.16 0.42 0.86],'LineWidth',22);
            harm = plot(ax, [0 0],[0.70 2.70],'-','Color',[0.92 0.78 0.20],'LineWidth',4);
            hbob = scatter(ax, 0, 2.70, 700, [1.0 0.28 0.16],'filled');
            ht_t = text(ax,-4.6,4.65,'T = 0.00 s',    'Color',[0.60 0.82 1.0],'FontSize',10,'FontName','Courier New');
            ht_x = text(ax,-4.6,4.22,'x = 0.000 m',   'Color',[0.28 0.68 1.0],'FontSize',10,'FontName','Courier New');
            ht_a = text(ax,-4.6,3.79,'th= 0.000 rad',  'Color',[1.00 0.80 0.22],'FontSize',10,'FontName','Courier New');
            ht_u = text(ax,-4.6,3.36,'u = 0.000 N',   'Color',[0.78 0.42 1.00],'FontSize',10,'FontName','Courier New');

            skip = max(1, round(0.025/dt));   % ~40 fps cap
            app.setStatus('Running...', [0.22 0.88 0.38]);

            % Live loop
            for k = 1:N
                if app.StopFlag; break; end

                xc    = sol(k,1);
                theta = sol(k,3);
                u_k   = -K * (sol(k,:)' - wr);

                if mod(k,skip)==0 || k==1
                    xv = xc;   % no clamping — axis follows cart

                    % Auto-scale x-axis: keep cart centred with half-width of 5m
                    % but expand if pendulum tip would go out of view
                    hw  = max(5, abs(al) + 1.5);   % half-width
                    xlim(ax, [xv - hw, xv + hw]);

                    % Cart
                    set(hw_l,'XData',xv-0.45,'YData',0.25);
                    set(hw_r,'XData',xv+0.45,'YData',0.25);
                    set(hbdy,'XData',[xv-0.75, xv+0.75],'YData',[0.65 0.65]);

                    % Pendulum
                    tx = xv - al*sin(-theta);
                    ty = 0.70 - al*cos(-theta);
                    set(harm,'XData',[xv, tx],'YData',[0.70, ty]);
                    set(hbob,'XData', tx, 'YData', ty);

                    % Bob colour
                    aerr = abs(theta - wr(3));
                    if     aerr < 0.05; set(hbob,'CData',[0.10 0.90 0.25]);
                    elseif aerr < 0.25; set(hbob,'CData',[1.00 0.80 0.15]);
                    else;               set(hbob,'CData',[1.00 0.22 0.14]);
                    end

                    % Telemetry text
                    set(ht_t,'String',sprintf('T  = %.2f s',   t_sol(k)));
                    set(ht_x,'String',sprintf('x  = %.3f m',   xc));
                    set(ht_a,'String',sprintf('th = %.4f rad',  theta));
                    set(ht_u,'String',sprintf('u  = %.2f N',   u_k));

                    % Stream state plots
                    idx = 1:k;
                    for i = 1:4
                        set(hLines{i},'XData',t_sol(idx),'YData',sol(idx,i));
                    end
                    drawnow limitrate;
                end
            end

            if app.StopFlag
                app.setStatus('Stopped.', [1.0 0.55 0.20]);
            else
                app.setStatus(sprintf('Done.   x=%.3f m   theta=%.4f rad', ...
                    sol(end,1), sol(end,3)), [0.22 0.88 0.38]);
            end
            app.BtnRun.Enable = 'on';
        end

        % ── Prep animation axes ─────────────────────────────────────
        function setupAnimAxes(app, wr, al)
            ax = app.AxAnim;
            cla(ax); hold(ax,'on'); grid(ax,'on');
            top = max(5, al+1.5);
            xlim(ax,[-5 5]); ylim(ax,[0 top]);
            xlabel(ax,'Position (m)','Color',[0.42 0.52 0.62]);
            title(ax,'Live Animation — Cart & Pendulum', ...
                  'Color',[0.65 0.82 1.0],'FontSize',11,'FontName','Courier New');
            % Target marker (stays fixed in world coords — xline auto-follows view)
            xline(ax, wr(1),'--','Color',[0.18 0.60 0.28 0.40],'LineWidth',1.2);
            % Ground line spans full world (auto-visible as view scrolls)
            plot(ax,[-1e4 1e4],[0 0],       '-','Color',[0.28 0.38 0.48],'LineWidth',2);
            plot(ax,[-1e4 1e4],[0.28 0.28], '-','Color',[0.16 0.24 0.34],'LineWidth',5);
        end

        function setStatus(app, msg, col)
            app.LblStatus.Text = msg;
            app.LblStatus.FontColor = col;
        end

        function stopSimulation(app)
            app.StopFlag = true;
        end

    end

    % ================================================================
    %  BUILD UI
    % ================================================================
    methods (Access = private)

        function buildUI(app)
            W = 1400; H = 860;

            % Colours
            BG  = [0.05 0.07 0.12];
            PAN = [0.07 0.10 0.16];
            BDR = [0.12 0.17 0.27];
            TXT = [0.78 0.88 0.96];
            ACC = [0.25 0.55 1.00];
            EF  = [0.09 0.13 0.21];
            FNT = 'Courier New';

            % Figure
            app.UIFigure = uifigure('Visible','off');
            app.UIFigure.Position = [40 40 W H];
            app.UIFigure.Name     = 'Inverted Pendulum - LQR Simulator';
            app.UIFigure.Color    = BG;
            app.UIFigure.Resize   = 'off';
            app.UIFigure.CloseRequestFcn = @(~,~) delete(app);

            % Title bar
            tb = uilabel(app.UIFigure,'Position',[0 H-40 W 40]);
            tb.Text = '  INVERTED PENDULUM  -  LQR CONTROLLER  -  Interactive Simulator';
            tb.FontName=FNT; tb.FontSize=13; tb.FontWeight='bold';
            tb.FontColor=ACC; tb.BackgroundColor=[0.03 0.05 0.09];

            % ── LEFT PANEL ──────────────────────────────────────────
            LW = 292;
            app.PanelLeft = uipanel(app.UIFigure,'Position',[6 6 LW H-48]);
            app.PanelLeft.BackgroundColor=PAN;
            app.PanelLeft.BorderColor=BDR;
            app.PanelLeft.Title='';

            P = app.PanelLeft;

            % Helper to make a label
            function h = mk_lbl(txt,x,y,w)
                h = uilabel(P,'Text',txt,'Position',[x y w 19],...
                    'FontName',FNT,'FontSize',9,'FontColor',TXT,...
                    'BackgroundColor',PAN);
            end
            % Helper to make a numeric edit field
            function h = mk_ef(val,x,y,w)
                h = uieditfield(P,'numeric','Value',val,'Position',[x y w 22],...
                    'FontName',FNT,'FontSize',9,...
                    'FontColor',[0.90 0.95 1.0],'BackgroundColor',EF);
            end
            % Helper for section header
            function mk_sec(txt,y)
                uilabel(P,'Text',['  ' txt],...
                    'Position',[4 y LW-10 19],...
                    'FontName',FNT,'FontSize',8,'FontWeight','bold',...
                    'FontColor',[0.35 0.60 0.95],...
                    'BackgroundColor',[0.055 0.085 0.14]);
            end

            y = H - 90;

            % SYSTEM PARAMETERS
            mk_sec('SYSTEM PARAMETERS', y); y = y-25;
            mk_lbl('Pendulum mass  (kg)', 8, y, 160); app.EditPendMass = mk_ef(50,   175, y, 105); y=y-25;
            mk_lbl('Cart mass      (kg)', 8, y, 160); app.EditCartMass = mk_ef(10,   175, y, 105); y=y-25;
            mk_lbl('Arm length     (m)',  8, y, 160); app.EditArmLen   = mk_ef(3,    175, y, 105); y=y-25;
            mk_lbl('Gravity      (m/s2)', 8, y, 160); app.EditGravity  = mk_ef(-9.81,175, y, 105); y=y-25;
            mk_lbl('Friction  d',         8, y, 160); app.EditFriction = mk_ef(1,    175, y, 105); y=y-30;

            % LQR WEIGHTS
            mk_sec('LQR WEIGHTS   Q=diag([Q1..Q4]),  R', y); y=y-25;
            mk_lbl('Q1  cart position', 8, y, 160); app.EditQ1 = mk_ef(1,      175, y, 105); y=y-25;
            mk_lbl('Q2  cart velocity', 8, y, 160); app.EditQ2 = mk_ef(1,      175, y, 105); y=y-25;
            mk_lbl('Q3  angle',         8, y, 160); app.EditQ3 = mk_ef(1,      175, y, 105); y=y-25;
            mk_lbl('Q4  angular rate',  8, y, 160); app.EditQ4 = mk_ef(1,      175, y, 105); y=y-25;
            mk_lbl('R   control cost',  8, y, 160); app.EditR  = mk_ef(0.0001, 175, y, 105); y=y-30;

            % TARGET STATE
            mk_sec('TARGET STATE   wr = [x, 0, theta, 0]', y); y=y-25;
            mk_lbl('Cart position  (m)', 8, y, 160); app.EditTx     = mk_ef(0,  175, y, 105); y=y-25;
            mk_lbl('Angle theta    (rad)',8, y, 160); app.EditTtheta = mk_ef(pi, 175, y, 105); y=y-30;

            % INITIAL STATE
            mk_sec('INITIAL STATE   x0', y); y=y-25;
            mk_lbl('Position  x0  (m)',   8, y, 160); app.EditIx     = mk_ef(-2,    175, y, 105); y=y-25;
            mk_lbl('Velocity  v0  (m/s)', 8, y, 160); app.EditIv     = mk_ef(0,     175, y, 105); y=y-25;
            mk_lbl('Angle    th0  (rad)', 8, y, 160); app.EditItheta = mk_ef(pi+0.1,175, y, 105); y=y-25;
            mk_lbl('Ang.rate  w0',        8, y, 160); app.EditIomega = mk_ef(0,     175, y, 105); y=y-30;

            % TIME
            mk_sec('TIME SETTINGS', y); y=y-25;
            mk_lbl('Duration   (s)',  8, y, 160); app.EditTend = mk_ef(20,   175, y, 105); y=y-25;
            mk_lbl('Time step  dt',   8, y, 160); app.EditDt   = mk_ef(0.01, 175, y, 105); y=y-38;

            % Run / Stop buttons
            app.BtnRun = uibutton(P,'push',...
                'Position',[6 y-4 136 36],...
                'Text','Run Simulation',...
                'FontName',FNT,'FontSize',11,'FontWeight','bold',...
                'BackgroundColor',[0.12 0.38 0.80],'FontColor',[1 1 1],...
                'ButtonPushedFcn',@(~,~) app.runSimulation());

            app.BtnStop = uibutton(P,'push',...
                'Position',[148 y-4 130 36],...
                'Text','Stop',...
                'FontName',FNT,'FontSize',11,...
                'BackgroundColor',[0.52 0.12 0.12],'FontColor',[1 1 1],...
                'ButtonPushedFcn',@(~,~) app.stopSimulation());

            y = y - 48;

            % Status label
            app.LblStatus = uilabel(P,...
                'Position',[6 y LW-14 20],...
                'Text','Ready.  Press Run to start.',...
                'FontName',FNT,'FontSize',8,...
                'FontColor',[0.40 0.75 0.40],'BackgroundColor',PAN);

            y = y - 26;

            % Gains display
            app.LblGains = uilabel(P,...
                'Position',[6 y-8 LW-14 30],...
                'Text','K = (computed after run)',...
                'FontName',FNT,'FontSize',7,'WordWrap','on',...
                'FontColor',[0.42 0.60 0.90],'BackgroundColor',PAN);

            % ── RIGHT PANEL ─────────────────────────────────────────
            RX = LW+14;
            RW = W-RX-6;

            app.PanelRight = uipanel(app.UIFigure,'Position',[RX 6 RW H-48]);
            app.PanelRight.BackgroundColor=PAN;
            app.PanelRight.BorderColor=BDR;
            app.PanelRight.Title='';

            axBG = [0.04 0.06 0.10];
            axXC = [0.32 0.42 0.52];
            axGC = [0.09 0.13 0.20];

            % Animation axes — top 55%
            animH = round((H-54)*0.55);
            app.AxAnim = uiaxes(app.PanelRight,...
                'Position',[6 (H-54)-animH-2 RW-14 animH-4]);
            app.AxAnim.Color=axBG; app.AxAnim.XColor=axXC; app.AxAnim.YColor=axXC;
            app.AxAnim.GridColor=axGC; app.AxAnim.GridAlpha=0.55;
            app.AxAnim.XGrid='on'; app.AxAnim.YGrid='on'; app.AxAnim.Box='on';

            % 4 state plots — bottom 41%
            plotH = round((H-54)*0.41);
            plotW = floor((RW-28)/4);
            axHandles = cell(4,1);
            for i = 1:4
                axHandles{i} = uiaxes(app.PanelRight,...
                    'Position',[6+(i-1)*(plotW+5) 6 plotW plotH-4]);
                axHandles{i}.Color=axBG;
                axHandles{i}.XColor=axXC; axHandles{i}.YColor=axXC;
                axHandles{i}.GridColor=axGC; axHandles{i}.GridAlpha=0.5;
                axHandles{i}.XGrid='on'; axHandles{i}.YGrid='on';
            end
            app.AxX     = axHandles{1};
            app.AxV     = axHandles{2};
            app.AxTheta = axHandles{3};
            app.AxOmega = axHandles{4};

            app.UIFigure.Visible = 'on';
        end
    end

    % ================================================================
    %  CONSTRUCTOR / DESTRUCTOR
    % ================================================================
    methods (Access = public)

        function app = InvertedPendulumApp()
            buildUI(app);
            registerApp(app, app.UIFigure);
            if nargout == 0
                clear app
            end
        end

        function delete(app)
            if isvalid(app.UIFigure)
                delete(app.UIFigure);
            end
        end
    end

end


% ================================================================
%  NONLINEAR ODE  (ode45 can access it)
% ================================================================
function dxdt = pendcart_ode(x, K, wr, pm, cm, al, gv, dv)
    x   = x(:);
    Sx  = sin(x(3));
    Cx  = cos(x(3));
    Det = pm * al^2 * (pm + cm*(1 - Cx^2));
    u   = -K * (x - wr);

    x2d = (1/Det)*(-(pm^2)*al^2*gv*Cx*Sx ...
          + pm*al^2*(pm*al*(x(4)^2)*Sx - dv*x(2))) ...
          + pm*al^2*(1/Det)*u;

    x4d = (1/Det)*((pm+cm)*pm*gv*al*Sx ...
          - pm*al*Cx*(pm*al*(x(4)^2)*Sx - dv*x(2))) ...
          - pm*al*Cx*(1/Det)*u;

    dxdt = [x(2); x2d; x(4); x4d];
end